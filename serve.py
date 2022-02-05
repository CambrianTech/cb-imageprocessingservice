import asyncio
from concurrent.futures import ThreadPoolExecutor
from os.path import join
import os
import json
import datetime
import dateutil
from time import time
import typing
import functools

import click
from aiohttp import web
import aiohttp_cors
import boto3
import requests
from gpuinfo import GPUInfo

from pipeline.pipeline import Pipeline, PipelineMode

def _get_instance_metadata():
    metadata = {}

    # https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-instance-metadata.html
    try:
        req = requests.get(
            "http://169.254.169.254/latest/dynamic/instance-identity/document")
        response_json = req.json()
        metadata["region"] = response_json.get("region")
        metadata["instance_id"] = response_json.get("instanceId")
    except:
        return None

    # https://docs.aws.amazon.com/AmazonECS/latest/developerguide/container-metadata.html
    container_metadata_file_path = os.environ["ECS_CONTAINER_METADATA_FILE"]
    if container_metadata_file_path is None:
        return None

    with open(container_metadata_file_path, "r") as container_metadata_file:
        container_metadata = json.load(container_metadata_file)

    metadata["cluster"] = container_metadata["Cluster"]
    return metadata


@click.command()
@click.argument("model_path", type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.argument("semantic_model_path", type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.argument("fov_model_path", type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.argument("hed_model_path", default='hed_model/HED_pretrained_bsds.npz', type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.argument("user_uploads_bucket", type=click.STRING)
@click.argument("results_bucket", type=click.STRING)
@click.argument("plane_url", type=click.STRING)
@click.option("--sqs-queue-name", type=click.STRING, default=None)
@click.option('--api', type=int, default=3, help='api level: 1-4')
@click.option("--logging_dir", type=click.Path(exists=False, file_okay=False, dir_okay=True), default='logging')
@click.option('--log_level', type=int, default=0, help='corresponds to LogLevel inside pipeline/logging, a binary mask: models | segmentation | images, default All')
@click.option('--log_step', type=int, default=None, help='Log only a single step in the pipeline')
def main(model_path, semantic_model_path, fov_model_path, hed_model_path, user_uploads_bucket, results_bucket, plane_url, sqs_queue_name,
         api, logging_dir, log_level, log_step):

    metadata = _get_instance_metadata()
    
    #print("Instance metadata:", metadata)

    print("Setting default executor")
    asyncio.get_event_loop().set_default_executor(ThreadPoolExecutor())

    print("Creating pipeline")

    pipeline = Pipeline(PipelineMode.Serve, api, 
        src_path=user_uploads_bucket, 
        dest_path=results_bucket,
        model_path=model_path, 
        semantic_model_path=semantic_model_path, 
        fov_model_path=fov_model_path, 
        hed_model_path=hed_model_path,
        planes_url=plane_url, 
        logging_dir=logging_dir, 
        logging_level=log_level, 
        logging_step=log_step
        )

    pipeline.start()

    total_pipeline_times = []

    # Pipeline for finding planes, generating lighting and predicting fov.
    async def planes_pipeline(input_dict: typing.Dict):
        total_start_time = time()
        results = await pipeline.process(input_dict)
        total_pipeline_time = time() - total_start_time
        print("Planes total pipeline time: %.2fs" % total_pipeline_time)
        total_pipeline_times.append((total_pipeline_time, datetime.datetime.now(dateutil.tz.tzlocal())))
        return results

    # Setup http server
    def get_pipeline_handler(pipeline_fn):
        async def handle(request):
            print("Handle segment:", request, "(items waiting in pipeline: %d)" %
                  num_waiting_items(pipeline.steps))

            # Get image S3 key from GET request
            unique_id = request.match_info.get("id", None)
            if unique_id is None:
                raise web.HTTPBadRequest()

            data = {"unique_id": unique_id}

            data = await pipeline_fn(data)

            response_dict = {
                "lighting_url": data["lighting_url"],
                "semantic_url": data["semantic_url"],
                "data_url": data["data_v3_url"],
                "superpixels_url": data["superpixels_url"],
            }

            if "data_v2_url" in data:
                response_dict["data_v2_url"] = data["data_v2_url"]

            if "data_v3_url" in data:
                response_dict["data_v3_url"] = data["data_v3_url"]

            return web.json_response(response_dict)
        return handle

    print("SQS Queue name:", sqs_queue_name)
    if sqs_queue_name is not None:
        if metadata is None:
            raise Exception("sqs_queue name was set but metadata was none")
        
        async def sqs_loop():
            loop = asyncio.get_event_loop()

            print("Getting SQS queue with name", sqs_queue_name)
            sqs = boto3.resource("sqs", region_name=metadata["region"])
            sqs_queue = sqs.get_queue_by_name(QueueName=sqs_queue_name)

            def get_new_messages():
                return sqs_queue.receive_messages(MaxNumberOfMessages=1, WaitTimeSeconds=10)

            print("Starting to ingest SQS messages")
            while True:
                try:
                    msgs = await loop.run_in_executor(None, get_new_messages)
                except Exception as e:
                    print("Failed to receive messages from SQS queue:", e)
                    continue
                for msg in msgs:
                    print("Received SQS message:", msg)
                    await loop.run_in_executor(None, msg.delete)
                    try:
                        print("Loading SQS message")
                        msg_data = json.loads(msg.body)
                        print("Running message in pipeline", msg_data)
                        result_data = await planes_pipeline(msg_data)
                        print("Got pipeline results", result_data)
                    except Exception as e:
                        print("Error processing SQS message:", e)
        print("Starting SQS loop")
        asyncio.ensure_future(sqs_loop())

    async def handle_healthcheck(request):
        return web.Response(text="Healthy")

    async def push_metrics_loop():
        loop = asyncio.get_event_loop()

        if metadata is not None:
            cw = boto3.client("cloudwatch", region_name=metadata["region"])

        def push_metrics():
            metric_time = datetime.datetime.now(dateutil.tz.tzlocal())

            metric_data = []

            for total_pipeline_time, total_pipeline_time_timestamp in total_pipeline_times:
                metric_data.append({
                    "MetricName": "PipelineTotalTime",
                    "Dimensions": [{
                        "Name": "ClusterName",
                        "Value": metadata["cluster"]
                    }],
                    "Timestamp": total_pipeline_time_timestamp,
                    "Value": total_pipeline_time
                })
            total_pipeline_times.clear()

            gpu_usages, gpu_memories = GPUInfo.gpu_usage()
            print("GPU Usages:", gpu_usages, "memories:", gpu_memories)
            if len(gpu_usages) > 0:
                for gpu_index, (gpu_usage, gpu_memory) in enumerate(zip(gpu_usages, gpu_memories)):
                    metric_data.append({
                        "MetricName": "GPUUsage%d" % gpu_index,
                        "Dimensions": [{
                            "Name": "ClusterName",
                            "Value": metadata["cluster"]
                        }],
                        "Timestamp": metric_time,
                        "Value": gpu_usage
                    })

                    metric_data.append({
                        "MetricName": "GPUMemory%d" % gpu_index,
                        "Dimensions": [{
                            "Name": "ClusterName",
                            "Value": metadata["cluster"]
                        }],
                        "Timestamp": metric_time,
                        "Value": gpu_memory
                    })
            else:
                metric_data.append({
                    "MetricName": "GPUUsage0",
                    "Dimensions": [{
                        "Name": "ClusterName",
                        "Value": metadata["cluster"]
                    }],
                    "Timestamp": metric_time,
                    "Value": 0
                })

                metric_data.append({
                    "MetricName": "GPUMemory0",
                    "Dimensions": [{
                        "Name": "ClusterName",
                        "Value": metadata["cluster"]
                    }],
                    "Timestamp": metric_time,
                    "Value": 0
                })

            cw.put_metric_data(
                Namespace="ImageProcessingService",
                MetricData=metric_data
            )

        # Publish metrics to CloudWatch every minute
        while True:
            await asyncio.sleep(60)
            if metadata is not None:
                await loop.run_in_executor(None, push_metrics)


    print("Starting metrics loop")
    asyncio.ensure_future(push_metrics_loop())

    print("Creating web app")
    app = web.Application()

    cors = aiohttp_cors.setup(app, defaults={
        "*": aiohttp_cors.ResourceOptions(
            allow_credentials=True,
            expose_headers="*",
            allow_headers="*",
        )
    })

    # Add private (non-CORS) routes
    app.add_routes([
        web.get("/healthcheck", handle_healthcheck)
    ])

    print("Running web app")
    web.run_app(app)


if __name__ == "__main__":
    main()

import asyncio
import subprocess
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

from pipeline.core import schedule_and_wait, merge_future_dicts, num_waiting_items
from pipeline.fov import PipelineCalculateFov
from pipeline.getdata import PipelineGetData
from pipeline.primaryangle import PipelineDeterminePrimaryAngles
# from pipeline.refine import PipelineRefineResults
from pipeline.runmodels import PipelineRunModels
from pipeline.superpixels import PipelineSuperpixels
from pipeline.refineplanemasks import PipelineRefinePlaneMasks
from pipeline.combineplanemasks import PipelineCombinePlaneMasks
from pipeline.uploadresults import PipelineUploadResults
from pipeline.remote import PipelineRemotePlaneDetector, PipelineRemoteNetworks


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


default_config = PipelineConfig()

@click.command()
@click.argument("model_path", default=default_config.model_path, type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.argument("semantic_model_path", default=default_config.semantic_model_path, type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.argument("fov_model_path", default=default_config.fov_model_path, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.argument("hed_model_path", default=default_config.hed_model_path, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.argument("user_uploads_bucket", type=click.STRING)
@click.argument("results_bucket", type=click.STRING)
@click.argument("planes_url", default=default_config.planes_url, type=click.STRING)
@click.option("--image-local-dir", type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option("--results-local-dir", type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option("--sqs-queue-name", type=click.STRING, default=None)
@click.option('--api', type=int, default=default_config.api_level, help='api level: 1-4')
@click.option("--log_dir", type=click.Path(exists=False, file_okay=False, dir_okay=True), default=None)
@click.option('--log_level', type=int, default=default_config.logging_level, help='corresponds to LogLevel inside pipeline/logging, a binary mask: models | segmentation | images, default All')
@click.option('--log_step', type=int, default=default_config.logging_step, help='Log only a single step in the pipeline')
def main(model_path, semantic_model_path, fov_model_path, hed_model_path, user_uploads_bucket, results_bucket, planes_url, 
        image_local_dir, results_local_dir, sqs_queue_name, api, log_dir, log_level, log_step):

    if results_local_dir is None:
        print("Trying to get instance metadata")
        metadata = _get_instance_metadata()
    else:
        metadata = None
        print("Working locally. Metadata set to None")

    if metadata is not None:
        print("Instance metadata:", metadata)
    
    print("Setting default executor")
    asyncio.get_event_loop().set_default_executor(ThreadPoolExecutor())

    print("Starting CPU networks process")
    cpu_networks_port = 8082
    subprocess.Popen(["python3", "runcpunetworks.py",
                      model_path, str(cpu_networks_port)])

    print("Creating pipeline")

    config = PipelineConfig()
    config.mode = PipelineMode.Serve
    config.src_path=user_uploads_bucket
    config.dest_path=results_bucket

    config.model_path = model_path
    config.semantic_model_path = semantic_model_path
    config.fov_model_path = fov_model_path
    config.hed_model_path = hed_model_path
    config.planes_url = planes_url
    config.sqs_queue_name = sqs_queue_name

    config.api_level = api
    config.logging_dir = log_dir
    config.logging_level = log_level
    config.logging_step = None if log_step is None else PipelineStepIndex(log_step)

    pipeline = Pipeline(config)

    # Start the processing workers for all steps
    pipeline.start()

    total_pipeline_times = []

    # Pipeline for finding planes, generating lighting and predicting fov.
    async def planes_pipeline(input_dict: typing.Dict):

        if 'image_s3_key' in input_dict:
            input_dict['unique_id'] = input_dict['image_s3_key']
            print("Warning 'image_s3_key' is no longer being used. Please update this to 'unique_id'")

        print("Running image %s through pipeline" % input_dict['unique_id'])

        total_start_time = time()

        results = await pipeline.process(input_dict)

        total_pipeline_time = time() - total_start_time
        print("Planes total pipeline time: %.2fs" % total_pipeline_time)
        total_pipeline_times.append((total_pipeline_time, datetime.datetime.now(dateutil.tz.tzlocal())))
        return results

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
                        print("Got pipeline results")
                    except Exception as e:
                        print("Error processing SQS message:", e)
        print("Starting SQS loop")
        asyncio.ensure_future(sqs_loop())
        
    async def handle_healthcheck(request):
        return web.Response(text="Healthy")

    async def handle_local_upload(request):
        print("Handle local file upload.", request)

        unique_id = request.match_info.get("id", None)
        if unique_id is None:
            raise web.HTTPBadRequest()

        output_dir = join(image_local_dir, user_uploads_bucket)
        os.makedirs(output_dir, exist_ok=True)

        # Read 1MB chunks into the file
        with open(join(output_dir, unique_id), "wb") as image_file:
            while True:
                chunk = await request.content.read(1024*1024)
                if not chunk:
                    break
                image_file.write(chunk)

        return web.json_response({})

    async def handle_get_image(request):
        print("Handle get image:", request)

        unique_id = request.match_info.get("id", None)
        bucket = request.match_info.get("bucket", None)
        if unique_id is None or bucket is None:
            raise web.HTTPBadRequest()

        return web.FileResponse(os.path.join(results_local_dir, bucket, unique_id))

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


    # Add endpoint for directly getting and uploading images if local
    # image input dir was defined
    if image_local_dir is not None:
        upload_resource = app.router.add_resource("/upload/{id}")
        cors.add(upload_resource.add_route("PUT", handle_local_upload))

        get_image_resource = app.router.add_resource("/getimage/{bucket}/{id}")
        cors.add(get_image_resource.add_route("GET", handle_get_image))

    # Add private (non-CORS) routes
    app.add_routes([
        web.get("/healthcheck", handle_healthcheck)
    ])

    print("Running web app")
    web.run_app(app)


if __name__ == "__main__":
    main()

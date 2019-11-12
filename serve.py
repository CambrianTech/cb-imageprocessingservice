import asyncio
from concurrent.futures import ThreadPoolExecutor
from os.path import join
import os
import json
import datetime
import dateutil

import click
from aiohttp import web
import aiohttp_cors
import boto3
import requests

from pipeline.core import Pipeline
from pipeline.fov import PipelineCalculateFov
from pipeline.getdata import PipelineGetData
from pipeline.primaryangle import PipelineDeterminePrimaryAngles
from pipeline.refine import PipelineRefineResults
from pipeline.runmodels import PipelineRunModels
from pipeline.superpixels import PipelineSuperpixels
from pipeline.uploadresults import PipelineUploadResults


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
@click.argument("fov_model_path", type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.argument("user_uploads_bucket", type=click.STRING)
@click.argument("results_bucket", type=click.STRING)
@click.option("--image-local-dir", type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option("--results-local-dir", type=click.Path(exists=True, file_okay=False, dir_okay=True))
def main(model_path, fov_model_path, user_uploads_bucket, results_bucket, image_local_dir, results_local_dir):
    print("Setting default executor")
    asyncio.get_event_loop().set_default_executor(ThreadPoolExecutor())

    print("Creating pipeline")

    # Setup pipeline to run on requests
    pipeline = (Pipeline()
                .add(PipelineGetData(user_uploads_bucket))
                .add(PipelineRunModels(
                    semantic_path=join(model_path, "semantic"),
                    normals_path=join(model_path, "normals"),
                    unlit_path=join(model_path, "unlit"),
                    elevation_path=join(model_path, "elevation"),
                    lighting_path=join(model_path, "lighting")
                ))
                .add(PipelineDeterminePrimaryAngles())
                .add(PipelineRefineResults())
                .add(PipelineCalculateFov(fov_model_path))
                .add(PipelineSuperpixels())
                .add(PipelineUploadResults(results_bucket)))

    print("Validating pipeline")
    pipeline.validate(["image_s3_key"])

    # Setup http server
    async def handle_segment(request):
        print("Handle segment:", request, "(items waiting in pipeline: %d)" %
              pipeline.num_waiting_items)

        # Get image S3 key from GET request
        image_s3_key = request.match_info.get("id", None)
        if image_s3_key is None:
            raise web.HTTPBadRequest()

        data = {"image_s3_key": image_s3_key}

        # Add local directories to initial data if specified
        if image_local_dir is not None:
            print(
                "WARNING: Do not use in production: image local dir set to", image_local_dir)
            data["image_local_dir"] = image_local_dir
        if results_local_dir is not None:
            print(
                "WARNING: Do not use in production: results local dir set to", results_local_dir)
            data["results_local_dir"] = results_local_dir

        data = await pipeline.run(data)

        return web.json_response({
            "lighting_url": data["lighting_url"],
            "semantic_url": data["semantic_url"],
            "data_url": data["data_url"],
            "superpixels_url": data["superpixels_url"],
        })

    async def handle_healthcheck(request):
        return web.Response(text="Healthy")

    async def handle_local_upload(request):
        print("Handle local file upload", request)

        image_s3_key = request.match_info.get("id", None)
        if image_s3_key is None:
            raise web.HTTPBadRequest()

        output_dir = join(image_local_dir, user_uploads_bucket)
        os.makedirs(output_dir, exist_ok=True)

        # Read 1MB chunks into the file
        with open(join(output_dir, image_s3_key), "wb") as image_file:
            while True:
                chunk = await request.content.read(1024*1024)
                if not chunk:
                    break
                image_file.write(chunk)
                
        return web.json_response({})

    async def handle_get_image(request):
        print("Handle get image:", request)

        image_s3_key = request.match_info.get("id", None)
        bucket = request.match_info.get("bucket", None)
        if image_s3_key is None or bucket is None:
            raise web.HTTPBadRequest()

        return web.FileResponse(os.path.join(results_local_dir, bucket, image_s3_key))

    print("Trying to get instance metadata")
    
    metadata = None
    
    if results_local_dir is None:
        metadata = _get_instance_metadata()
    else:
        metadata = None
        print("Working locally. Metadata set to None")

    if metadata is not None:
        print("Instance metadata:", metadata)

    async def push_metrics_loop():
        loop = asyncio.get_event_loop()

        if metadata is not None:
            cw = boto3.client("cloudwatch", region_name=metadata["region"])

        def push_metrics(avg_waiting_items):
            cw.put_metric_data(Namespace="ImageProcessingService",
                               MetricData=[{
                                   "MetricName": "PipelineWaitingItems",
                                   "Dimensions": [{
                                       "Name": "ClusterName",
                                       "Value": metadata["cluster"]
                                   }],
                                   "Timestamp": datetime.datetime.now(
                                       dateutil.tz.tzlocal()),
                                   "Value": avg_waiting_items
                               }])

        avg_waiting_items = 0

        # Publish metrics to CloudWatch every minute
        while True:
            for _ in range(60):
                await asyncio.sleep(1)
                avg_waiting_items = 0.9 * avg_waiting_items + 0.1 * pipeline.num_waiting_items

            print("Waiting items: %.2f (avg: %.2f)" %
                  (pipeline.num_waiting_items, avg_waiting_items))

            if metadata is not None:
                await loop.run_in_executor(None, push_metrics, avg_waiting_items)

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

    # Add public (CORS) routes
    segment_resource = app.router.add_resource("/segment/{id}")
    cors.add(segment_resource.add_route("GET", handle_segment))

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

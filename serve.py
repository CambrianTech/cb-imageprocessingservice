import asyncio
from concurrent.futures import ThreadPoolExecutor
from os.path import join

import click
from aiohttp import web

from pipeline.core import Pipeline
from pipeline.fov import PipelineCalculateFov
from pipeline.getdata import PipelineGetData
from pipeline.primaryangle import PipelineDeterminePrimaryAngles
from pipeline.refine import PipelineRefineResults
from pipeline.runmodels import PipelineRunModels
from pipeline.uploadresults import PipelineUploadResults


@click.command()
@click.argument("model_path", type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.argument("fov_model_path", type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.argument("user_uploads_bucket", type=click.STRING)
@click.argument("results_bucket", type=click.STRING)
def main(model_path, fov_model_path, user_uploads_bucket, results_bucket):
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
                .add(PipelineUploadResults(results_bucket)))

    print("Validating pipeline")
    pipeline.validate(["image_s3_key"])

    # Setup http server
    async def handle_segment(request):
        print("Handle segment:", request)

        # Get image S3 key from GET request
        image_s3_key = request.match_info.get("id", None)
        if image_s3_key is None:
            raise web.HTTPBadRequest()

        data = {"image_s3_key": image_s3_key}

        data = await pipeline.run(data)

        return web.json_response({
            "lighting_url": data["lighting_url"],
            "semantic_url": data["semantic_url"],
            "fov": data["fov"],
        })

    print("Creating web app")
    app = web.Application()
    app.add_routes(([
        web.get("/segment/{id}", handle_segment)
    ]))

    print("Running web app")
    web.run_app(app)


if __name__ == "__main__":
    main()

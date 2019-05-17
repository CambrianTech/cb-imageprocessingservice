import click
import asyncio
from aiohttp import web
from os.path import join

from pipeline.fov import PipelineCalculateFov
from pipeline.refine import PipelineRefineResults
from pipeline.runmodels import PipelineRunModels
from pipeline.primaryangle import PipelineDeterminePrimaryAngles
from pipeline.getdata import PipelineGetData
from pipeline.uploadresults import PipelineUploadResults
from pipeline.core import Pipeline


@click.command()
@click.argument("model_path", type=click.Path(exists=True, file_okay=False, dir_okay=True))
def main(model_path):
    # Setup pipeline to run on requests
    pipeline = (Pipeline()
                .add(PipelineGetData())
                .add(PipelineRunModels(
                    semantic_path=join(model_path, "semantic"),
                    normals_path=join(model_path, "normals"),
                    unlit_path=join(model_path, "unlit"),
                    elevation_path=join(model_path, "elevation"),
                    lighting_path=join(model_path, "lighting")
                ))
                .add(PipelineDeterminePrimaryAngles())
                .add(PipelineRefineResults())
                .add(PipelineCalculateFov("sklearn_models/fov_classifier_lc128.joblib"))
                .add(PipelineUploadResults()))

    pipeline.validate(["image_s3_key"])

    # Setup http server
    async def handle_segment(request):
        loop = asyncio.get_event_loop()

        # Get image S3 key from GET request
        image_s3_key = request.match_info.get("id", None)
        if image_s3_key is None:
            raise web.HTTPBadRequest()

        data = {"image_s3_key": image_s3_key}

        def process_request():
            return pipeline.run(data)

        data = await loop.run_in_executor(None, process_request)

        return web.json_response({
            "lighting_url": data["lighting_url"],
            "semantic_url": data["semantic_url"],
        })

    app = web.Application()
    app.add_routes(([
        web.get("/segment/{id}", handle_segment)
    ]))

    print("Running web app")
    web.run_app(app)


if __name__ == "__main__":
    main()

import os
import pickle
import sys

from aiohttp import web
import numpy as np
import click
from modelutils import feed_image_batched, load_model, get_session_config

@click.command()
@click.argument("model_path", type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.argument("port", type=click.IntRange(0, 65535))
@click.argument("message", default="SUCCESS", type=click.STRING)
def main(model_path, port, message):
    max_size = 100 * 1024 * 1024  # Max size to receive

    print("Loading models from", model_path)
    model_lighting = load_model(os.path.join(model_path, "lighting"), session_config=get_session_config(use_gpu=False))
    model_normals = load_model(os.path.join(model_path, "normals"), session_config=get_session_config(use_gpu=False))

    routes = web.RouteTableDef()

    print("Models successfully loaded from", model_path)

    @routes.post("/healthcheck")
    async def healthcheck(request):
        # print("Received healthcheck request:", request)
        try:
            return web.Response(body=message)
        except Exception as e:
            print("healthcheck request error:", e)
            return web.HTTPInternalServerError()

    @routes.post("/")
    async def index(request):
        print("Received request:", request)
        try:
            data = await request.read()

            # Load images as numpy array from received file.
            # Dimensions: [B, H, W, C]
            print("Loading data from request")
            images = pickle.loads(data)

            lighting = feed_image_batched(model_lighting, images)
            normals = feed_image_batched(model_normals, images)

            print("Obtained lighting and normals", lighting.shape, normals.shape)

            return web.Response(body=pickle.dumps({
                "lighting": lighting,
                "normals": normals
            }))
        except Exception as e:
            print("Error:", e)
        
        return web.HTTPInternalServerError()

    app = web.Application(client_max_size=max_size)
    app.add_routes(routes)

    print(message, file=sys.stderr) #SIGNAL SUCCESS

    print("Starting web app on port", port)
    web.run_app(app, port=port)

if __name__ == "__main__":
    main()

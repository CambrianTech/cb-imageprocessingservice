import pickle
from aiohttp import web
import numpy as np
import torch
from torch.utils.data import DataLoader
import cv2

from options import parse_args
from evaluate import PlaneRCNNDetector
from config import InferenceConfig
from datasets.evaluation_dataset import EvaluationDataset
from models.model import compose_image_meta

# TODO: Only import needed stuff
from utils import *
from datasets.plane_dataset import *


def load_sample(image, camera, options, config):
    # Image wasn't loaded with cv2.imread (but with eg. imageio.imread)
    # so need to convert to BGR which plane-rcnn uses.
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Create a data loader to load the single image
    dataset = EvaluationDataset(
        options, config, image_list=[image], camera=camera)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

    for sample in loader:
        return sample

    raise Exception("Data loader did not return sample")


def dict_torch_to_numpy(torch_dict):
    numpy_dict = {}
    for k, v in torch_dict.items():
        numpy_dict[k] = v.cpu().numpy() if isinstance(v, torch.Tensor) else v
    return numpy_dict


def main():
    max_size = 100 * 1024 * 1024  # Max size to receive

    print("Loading detector")
    options = parse_args()
    config = InferenceConfig(options)
    detector = PlaneRCNNDetector(options, config, modelType="final")

    routes = web.RouteTableDef()

    @routes.post("/")
    async def index(request):
        print("Received request:", request)
        try:
            data = await request.read()

            # Load images as numpy array from received file.
            # Dimensions: [B, H, W, C]
            input_dicts = pickle.loads(data)

            # Plane-RCNN needs the data as single images plus their metadata.
            # The metadata is a list of numbers.
            samples = [load_sample(input_dict["image"], input_dict["camera"], options, config)
                    for input_dict in input_dicts]

            with torch.no_grad():
                detections = [dict_torch_to_numpy(
                    detector.detect(sample)[0]) for sample in samples]

            return web.Response(body=pickle.dumps(detections))
        except Exception as e:
           print("Error:", e)

        return web.HTTPInternalServerError()

    app = web.Application(client_max_size=max_size)
    app.add_routes(routes)

    print("Starting web app")
    web.run_app(app, port=8081)


if __name__ == "__main__":
    main()

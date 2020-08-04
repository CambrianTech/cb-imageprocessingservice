import pickle
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


options = parse_args()
config = InferenceConfig(options)


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


def model_fn(model_dir):
    print("Loading detector")
    with torch.no_grad():
        detector = PlaneRCNNDetector(options, config, modelType="final")


def input_fn(request_body, request_content_type):
    if request_content_type == "application/python-pickle":
        # Load images as numpy array from received file.
        # Dimensions: [B, H, W, C]
        input_dicts = pickle.loads(request_body)

        # Plane-RCNN needs the data as single images plus their metadata.
        # The metadata is a list of numbers.
        samples = [
            load_sample(
                input_dict["image"],
                input_dict["camera"],
                options, config
            )
            for input_dict in input_dicts
        ]

        return samples
    else:
        raise ValueError("Unsupported content type", request_content_type)


def predict_fn(input_object, model):
    detector = model
    samples = input_object
    with torch.no_grad():
        detections = [
            dict_torch_to_numpy(detector.detect(sample)[0])
            for sample in samples
        ]
    return detections


def output_fn(prediction, content_type):
    if request_content_type == "application/python-pickle":
        return pickle.dumps(detections)
    raise ValueError("Unsupported content type", content_type)

import numpy as np
from scipy.special import softmax
import cv2
import tensorflow as tf
from pipeline.core import PipelineStep
import os
from time import time
from tensorpack import *
from tensorpack.tfutils import gradproc, optimizer
from tensorpack.tfutils.sesscreate import NewSessionCreator
from tensorpack.tfutils.summary import add_moving_summary, add_param_summary
from gluoncv.model_zoo import get_model
from gluoncv.data.transforms.presets.segmentation import test_transform
from gluoncv.data import batchify
from mxnet import image
import mxnet as mx

from .combineplanemasks import combine_plane_masks, combine_plane_clusters

from pipeline.core import PipelineStep, PipelineStepIndex

class PipelineSemanticSegmentation(PipelineStep):
    def __init__(self, semantic_path: str):
        super().__init__()
        
        print("PipelineSemanticSegmentation", "Initializing")
        self.mx_ctx = mx.gpu(0)
        self.model_semantic = get_model(
            "deeplab_resnest269_ade", pretrained=True,
            root=semantic_path, ctx=self.mx_ctx
        )
        print("PipelineSemanticSegmentation", "Initialization complete")

    @property
    def index(self) -> PipelineStepIndex:
        return PipelineStepIndex.Segmentation

    @property
    def required_keys(self) -> list:
        return ["image"]

    @property
    def output_keys(self) -> list:
        return ["semantic", "semantic_probs"]

    @property
    def is_batched(self) -> bool:
        return False

    def predict(self, images):

        # Numpy to mx, resize, test-transform, batch
        semantic_inputs = [
            test_transform(mx.img.resize_short(mx.nd.array(image, dtype=np.uint8), 480), self.mx_ctx)
            for image in images
        ]

        # Run semantic segmentation model
        # TODO: Batch properly?
        semantic_results = [
            self.model_semantic.predict(inp).asnumpy()
            for inp in semantic_inputs
        ]

        return semantic_results

    def run(self, data):

        print("PipelineSemanticSegmentation", "segment")

        t = time()

        if self.is_batched:
            images = [datum["image"] for datum in data]

            results = self.predict(images)

            # Store logit and softmaxed results
            for datum, result in zip(data, results):
                datum["semantic"] = result
                datum["semantic_probs"] = softmax(result[0], axis=0)
        else:
            images = [data["image"]]

            result = self.predict(images)[0]

            data["semantic"] = result
            data["semantic_probs"] = softmax(result[0], axis=0)

        print("PipelineSemanticSegmentation", "Semantic model took %.2f seconds" % (time() - t))

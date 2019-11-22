import numpy as np
from scipy.special import softmax
import cv2
from modelutils import feed_image_batched, feed_images_batched, load_model
from pipeline.core import PipelineStep
import os


class CropLayer(object):
    def __init__(self, params, blobs):
        # initialize our starting and ending (x, y)-coordinates of
        # the crop
        self.startX = 0
        self.startY = 0
        self.endX = 0
        self.endY = 0
    
    def getMemoryShapes(self, inputs):
        # the crop layer will receive two inputs -- we need to crop
        # the first input blob to match the shape of the second one,
        # keeping the batch size and number of channels
        (inputShape, targetShape) = (inputs[0], inputs[1])
        (batchSize, numChannels) = (inputShape[0], inputShape[1])
        (H, W) = (targetShape[2], targetShape[3])
        
        # compute the starting and ending crop coordinates
        self.startX = int((inputShape[3] - targetShape[3]) / 2)
        self.startY = int((inputShape[2] - targetShape[2]) / 2)
        self.endX = self.startX + W
        self.endY = self.startY + H
        
        # return the shape of the volume (we'll perform the actual
        # crop during the forward pass
        return [[batchSize, numChannels, H, W]]
    
    def forward(self, inputs):
        # use the derived (x, y)-coordinates to perform the crop
        return [inputs[0][:, :, self.startY:self.endY,
                          self.startX:self.endX]]


class PipelineRunModels(PipelineStep):
    layer_registered = False

    def __init__(self, semantic_path: str, normals_path: str, unlit_path: str,
                 elevation_path: str, lighting_path: str):
        self.model_semantic = load_model(semantic_path)
        self.model_normals = load_model(normals_path)
        self.model_unlit = load_model(unlit_path)
        self.model_elevation = load_model(elevation_path)
        self.model_lighting = load_model(lighting_path)

        proto_path = os.path.sep.join(["hed_model", "deploy.prototxt"])
        model_path = os.path.sep.join(["hed_model", "hed_pretrained_bsds.caffemodel"])
        self.net = cv2.dnn.readNetFromCaffe(proto_path, model_path)

        if not PipelineRunModels.layer_registered:
            PipelineRefineResults.layer_registered = True
            cv2.dnn_registerLayer("Crop", CropLayer)

    @property
    def required_keys(self) -> list:
        return ["image"]

    @property
    def output_keys(self) -> list:
        return ["image", "semantic", "semantic_probs", "normals", "elevation", "lighting", "normals_latents"]

    @property
    def is_batched(self) -> bool:
        return True

    def run(self, data):
        images = [datum["image"] for datum in data]
        
        def _run_single(model, key):
            results = feed_image_batched(model, images)
            for datum, result in zip(data, results):
                datum[key] = result

        _run_single(self.model_elevation, "elevation")
        _run_single(self.model_lighting, "lighting")
        _run_single(self.model_unlit, "unlit")
        
        semantic_input = [{ "image": datum["image"], "unlit": datum["unlit"] } for datum in data]
        semantic_results = [s["output"] for s in feed_images_batched(self.model_semantic, semantic_input)]
        for datum, result in zip(data, semantic_results):
            datum["semantic"] = result
            datum["semantic_probs"] = softmax(result/255, axis=-1)

        # Normals output with latents
        input_tensor = list(self.model_normals.feed_tensors.values())[0]
        latent_tensors = self.model_normals.graph.get_tensor_by_name(
            "generator/decoder_8/conv2d_transpose/BiasAdd:0")
        output_tensor = list(self.model_normals.fetch_tensors.values())[0]
        normals_latents, normals = self.model_normals.session.run([latent_tensors, output_tensor], feed_dict={
            input_tensor: [cv2.resize(datum["image"], (512, 512)).astype(np.float32)/255 for datum in data]})
        normals_latents = normals_latents[:, :, :, :128].reshape(len(data), 1, -1)

        for datum, nl, n in zip(data, normals_latents, normals):
            datum["normals"] = n
            datum["normals_latents"] = nl

        # OpenCV Model for HED
        images = [cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (1024, 1024)) for img in images]

        blob = cv2.dnn.blobFromImages(images, scalefactor=1.0, size=(1024,1024),
                                      mean=(104.00698793, 116.66876762, 122.67891434),
                                      swapRB=False, crop=True)
        self.net.setInput(blob)
        heds = self.net.forward()

        for datum, hed in zip(data, heds):
            datum["hed"] = hed
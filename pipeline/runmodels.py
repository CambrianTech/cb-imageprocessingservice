from pipeline.core import PipelineStep
from modelutils import load_model, feed_image, feed_images
import numpy as np
import cv2


def _softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)


class PipelineRunModels(PipelineStep):
    def __init__(self, semantic_path: str, normals_path: str, unlit_path: str,
                 elevation_path: str, lighting_path: str):
        self.model_semantic = load_model(semantic_path)
        self.model_normals = load_model(normals_path)
        self.model_unlit = load_model(unlit_path)
        self.model_elevation = load_model(elevation_path)
        self.model_lighting = load_model(lighting_path)

    @property
    def required_keys(self) -> list:
        return ["image"]

    @property
    def output_keys(self) -> list:
        return ["image", "semantic", "semantic_probs", "normals", "elevation", "lighting", "normals_latents"]

    def run(self, data):
        data["elevation"] = feed_image(self.model_elevation, data["image"])
        data["lighting"] = feed_image(self.model_lighting, data["image"])
        data["unlit"] = feed_image(self.model_unlit, data["image"])
        data["semantic_probs"] = feed_images(self.model_semantic, {
            "image": data["image"],
            "unlit": data["unlit"]
        })["output"]
        data["semantic"] = _softmax(data["semantic_probs"])

        # Normals output with latents
        input_tensor = list(self.model_normals.feed_tensors.values())[0]
        latent_tensors = self.model_normals.graph.get_tensor_by_name(
            "generator/decoder_8/conv2d_transpose/BiasAdd:0")
        output_tensor = list(self.model_normals.feed_tensors.values())[0]
        normals_latents, data["normals"] = self.model_normals.session.run([latent_tensors, output_tensor], feed_dict={
            input_tensor: [cv2.resize(data["image"], (512, 512)).astype(np.float32)/255]})
        data["normals_latents"] = normals_latents[:, :, :, :128].reshape(1, -1)

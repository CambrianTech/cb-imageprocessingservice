import numpy as np
from scipy.special import softmax
import cv2
from modelutils import feed_image_batched, feed_images_batched, load_model
from pipeline.core import PipelineStep


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

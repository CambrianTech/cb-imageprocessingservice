import unittest
from pipeline.uploadresults import PipelineUploadResults
from pipeline.getdata import PipelineGetData
from pipeline.fov import PipelineCalculateFov
from pipeline.runmodels import PipelineRunModels
from pipeline.superpixels import PipelineSuperpixels
import numpy as np
import asyncio
from pathlib import Path


class TestPipelineUploadResults(unittest.TestCase):
    def test_standard(self):
        step = PipelineUploadResults("cb-imageprocessingservice-results")

        semantic_probs = np.zeros((512, 512, 2))
        lighting = np.zeros((512, 512))

        image_s3_key = "TestPipelineUploadResults"

        data = {
            "image_s3_key": "TestPipelineUploadResults",
            "semantic_probs": semantic_probs,
            "lighting": lighting
        }

        step.run(data)

        self.assertIs(data["image_s3_key"], image_s3_key)
        self.assertIs(data["semantic_probs"], semantic_probs)
        self.assertIs(data["lighting"], lighting)

        self.assertIn("semantic_url", data)
        self.assertIn("lighting_url", data)


class TestPipelineGetData(unittest.TestCase):
    def test_standard(self):
        step = PipelineGetData("cb-user-image-uploads")

        image_s3_key = "iTInnsV7hXrEKdPWJY2vO5y7LJ9uOey8"

        data = {
            "image_s3_key": image_s3_key
        }

        step.run(data)

        self.assertIs(data["image_s3_key"], image_s3_key)
        self.assertIn("image", data)


class TestPipelineCalculateFov(unittest.TestCase):
    def test_standard(self):
        step = PipelineCalculateFov(
            Path("sklearn_models") / "fov_classifier_lc128.joblib"
        )

        normals_latents = np.zeros((1, 2048), np.float32)

        data = {
            "normals_latents": normals_latents
        }

        step.run(data)

        self.assertIs(data["normals_latents"], normals_latents)
        self.assertIn("fov", data)
        self.assertTrue(0 <= data["fov"] <= 360)


class TestPipelineRunModels(unittest.TestCase):
    def test_standard(self):
        hed_path = Path("hed_model") / "HED_pretrained_bsds.npz"
        semantic_path = Path("tensorflow_models") / "semantic"

        step = PipelineRunModels(
            semantic_path=semantic_path,
            hed_path=hed_path
        )

        self.assertTrue(step.is_batched)

        image = np.zeros((512, 512, 3))

        data = {"image": image}
        data_batch = [data]

        step.run(data_batch)

        self.assertIs(data["image"], image)
        self.assertIn("semantic", data)
        self.assertIn("normals", data)
        self.assertIn("unlit", data)
        self.assertIn("elevation", data)
        self.assertIn("lighting", data)
        self.assertIn("normals_latents", data)


class TestPipelineSuperpixels(unittest.TestCase):
    def test_standard(self):
        step = PipelineSuperpixels()

        image = np.concatenate([
            np.zeros((512, 256, 3), dtype=np.uint8),
            255 * np.ones((512, 256, 3), dtype=np.uint8)
        ], axis=1)

        data = {"image": image}

        step.run(data)

        self.assertIs(data["image"], image)
        self.assertIn("superpixels", data)


if __name__ == "__main__":
    unittest.main()

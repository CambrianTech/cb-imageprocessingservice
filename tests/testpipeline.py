import unittest
from pipeline.uploadresults import PipelineUploadResults
from pipeline.getdata import PipelineGetData
from pipeline.fov import PipelineCalculateFov
from pipeline.runmodels import PipelineRunModels
from pipeline.superpixels import PipelineSuperpixels
from pipeline.core import Pipeline
import numpy as np
import asyncio
from os.path import join


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
            join("sklearn_models", "fov_classifier_lc128.joblib"))

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
        model_path = "tensorflow_models"

        step = PipelineRunModels(
            semantic_path=join(model_path, "semantic"),
            normals_path=join(model_path, "normals"),
            unlit_path=join(model_path, "unlit"),
            elevation_path=join(model_path, "elevation"),
            lighting_path=join(model_path, "lighting"),
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


class TestPipelineChained(unittest.TestCase):
    def test_run_models_into_fov(self):
        model_path = "tensorflow_models"

        image = np.zeros((512, 512, 3))

        data = {
            "image": image
        }

        pipeline = (Pipeline()
                    .add(PipelineRunModels(
                        semantic_path=join(model_path, "semantic"),
                        normals_path=join(model_path, "normals"),
                        unlit_path=join(model_path, "unlit"),
                        elevation_path=join(model_path, "elevation"),
                        lighting_path=join(model_path, "lighting")))
                    .add(PipelineCalculateFov(join("sklearn_models", "fov_classifier_lc128.joblib"))))

        # Since the pipeline does not destroy its loops on exit (which doesn't matter since that will never happen in containers),
        # this will output some warnings about tasks still pending.
        result_data = asyncio.get_event_loop().run_until_complete(pipeline.run(data))

        self.assertIs(data, result_data)
        self.assertIs(data["image"], image)
        self.assertIn("semantic", data)
        self.assertIn("normals", data)
        self.assertIn("unlit", data)
        self.assertIn("elevation", data)
        self.assertIn("lighting", data)
        self.assertIn("normals_latents", data)
        self.assertIn("fov", data)
        self.assertTrue(0 <= data["fov"] <= 360)


class TestPipelineSuperpixels(unittest.TestCase):
    def test_standard(self):
        step = PipelineSuperpixels()

        image = np.zeros((512, 512, 3))

        data = {"image": image}

        step.run(data)

        self.assertIs(data["image"], image)
        self.assertIn("superpixels", data)


if __name__ == "__main__":
    unittest.main()

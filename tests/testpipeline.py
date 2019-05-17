import unittest
from pipeline.uploadresults import PipelineUploadResults
from pipeline.getdata import PipelineGetData
from pipeline.fov import PipelineCalculateFov
from pipeline.runmodels import PipelineRunModels
from pipeline.core import Pipeline
import numpy as np
from os.path import join


class TestPipelineUploadResults(unittest.TestCase):
    def test_standard(self):
        pipeline = PipelineUploadResults("cb-imageprocessingservice")

        semantic = np.zeros((512, 512))
        lighting = np.zeros((512, 512))

        image_s3_key = "TestPipelineUploadResults"

        data = {
            "image_s3_key": "TestPipelineUploadResults",
            "semantic": semantic,
            "lighting": lighting
        }

        pipeline.run(data)

        self.assertIs(data["image_s3_key"], image_s3_key)
        self.assertIs(data["semantic"], semantic)
        self.assertIs(data["lighting"], semantic)

        self.assertIn("semantic_url", data)
        self.assertIn("lighting_url", data)


class TestPipelineGetData(unittest.TestCase):
    def test_standard(self):
        pipeline = PipelineGetData("cb-user-image-uploads")

        image_s3_key = "iTInnsV7hXrEKdPWJY2vO5y7LJ9uOey8"

        data = {
            "image_s3_key": image_s3_key
        }

        pipeline.run(data)

        self.assertIs(data["image_s3_key"], image_s3_key)
        self.assertIn("image", data)


class TestPipelineCalculateFov(unittest.TestCase):
    def test_standard(self):
        pipeline = PipelineCalculateFov(
            join("sklearn_models", "fov_classifier_lc128.joblib"))

        normals_latents = np.zeros((1, 2048), np.float32)

        data = {
            "normals_latents": normals_latents
        }

        pipeline.run(data)

        self.assertIs(data["normals_latents"], normals_latents)
        self.assertIn("fov", data)
        self.assertTrue(0 <= data["fov"] <= 360)


class TestPipelineRunModels(unittest.TestCase):
    def test_standard(self):
        model_path = "tensorflow_models"

        pipeline = PipelineRunModels(
            semantic_path=join(model_path, "semantic"),
            normals_path=join(model_path, "normals"),
            unlit_path=join(model_path, "unlit"),
            elevation_path=join(model_path, "elevation"),
            lighting_path=join(model_path, "lighting"),
        )

        image = np.zeros((512, 512, 3))

        data = {
            "image": image
        }

        pipeline.run(data)

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

        pipeline.run(data)

        self.assertIs(data["image"], image)
        self.assertIn("semantic", data)
        self.assertIn("normals", data)
        self.assertIn("unlit", data)
        self.assertIn("elevation", data)
        self.assertIn("lighting", data)
        self.assertIn("normals_latents", data)
        self.assertIn("fov", data)
        self.assertTrue(0 <= data["fov"] <= 360)


if __name__ == "__main__":
    unittest.main()

import unittest
from pipeline.uploadresults import PipelineUploadResults
from pipeline.getdata import PipelineGetData
from pipeline.fov import PipelineCalculateFov
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
        pipeline = PipelineCalculateFov(join("sklearn_models", "fov_classifier_lc128.joblib"))

        normals_latents = np.zeros((1, 2048), np.float32)

        data = {
            "normals_latents": normals_latents
        }

        pipeline.run(data)

        self.assertIs(data["normals_latents"], normals_latents)
        self.assertIn("fov", data)
        self.assertTrue(0 <= data["fov"] <= 360)

if __name__ == "__main__":
    unittest.main()

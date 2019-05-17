import unittest
from pipeline.uploadresults import PipelineUploadResults
from pipeline.getdata import PipelineGetData
import numpy as np


class TestPipelineUploadResults(unittest.TestCase):
    def test_standard(self):
        pipeline = PipelineUploadResults("cb-imageprocessingservice")

        semantic = np.zeros((512, 512))
        lighting = np.zeros((512, 512))

        data = {
            "image_s3_key": "TestPipelineUploadResults",
            "semantic": semantic,
            "lighting": lighting
        }

        pipeline.run(data)

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

        self.assertEqual(data["image_s3_key"], image_s3_key)
        self.assertIn("image", data)


if __name__ == "__main__":
    unittest.main()

import unittest
from pipeline.uploadresults import PipelineUploadResults
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

if __name__ == "__main__":
    unittest.main()
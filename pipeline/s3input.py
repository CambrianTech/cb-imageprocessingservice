from pipeline.input import PipelineInput
from pipeline.s3Client import S3Client
try:
    from imageio import imread
except:
    from scipy.misc import imread

class PipelineS3Input(PipelineInput):
    def __init__(self, base_path, s3Client:S3Client):
        super().__init__(base_path)
        self.s3Client = s3Client

    def run(self, data):
        # Get image from S3 or local folder if local dir is set.
        data["image"] = self.s3Client.get_image_from_s3(self.s3_client, self.base_path, data["unique_id"])
        data["image"] = data["image"][:, :, :3]
from pipeline.stages.input import PipelineInput
from .s3client import S3Client

class PipelineS3Input(PipelineInput):

    def run(self, info: dict) -> dict:

        print("Getting image from s3", self.config.src_path, "ID", info["unique_id"])

        data = list()
        data["image"] = self.pipeline.s3_client.get_image_from_s3(self.config.src_path, info["unique_id"])
        data["image"] = data["image"][:, :, :3] #drop alpha channel

        return data
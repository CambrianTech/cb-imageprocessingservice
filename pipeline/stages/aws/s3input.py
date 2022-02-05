from pipeline.stages.input import PipelineInput
from .s3client import S3Client

class PipelineS3Input(PipelineInput):

    def run(self, data):
        # Get image from S3 or local folder if local dir is set.
        print("Getting image from s3", self.pipeline.src_path, "ID", data["unique_id"])
        data["image"] = self.pipeline.s3_client.get_image_from_s3(self.pipeline.src_path, data["unique_id"])
        data["image"] = data["image"][:, :, :3] #drop alpha channel
        print("Got image %s at shape" % data["unique_id"], data["image"].shape)
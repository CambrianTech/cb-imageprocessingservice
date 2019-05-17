from pipeline.core import PipelineStep
from PIL import Image
from io import BytesIO


def _get_image_from_s3(s3_client, key: str) -> np.ndarray:
    response = s3_client.Object("cb-imageprocessinguseruploads", key).get()
    image_data = response["Body"].read()
    return Image.open(BytesIO(image_data))


class PipelineGetData(PipelineStep):
    @property
    def required_keys(self) -> list:
        return []

    @property
    def output_keys(self) -> list:
        return []

    def run(self, data):
        # TODO
        pass

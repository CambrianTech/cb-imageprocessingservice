from pipeline.core import PipelineStep
import joblib


class PipelineCalculateFov(PipelineStep):
    def __init__(self, fov_model_path):
        super().__init__()
        self.classifier = joblib.load(fov_model_path)

    @property
    def required_keys(self) -> list:
        return ["normals_latents"]

    @property
    def output_keys(self) -> list:
        return ["fov"]

    def run(self, data):
        # Setfov.

        data["fov"] = 60.0

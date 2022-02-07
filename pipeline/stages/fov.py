from pipeline.core import PipelineStep, PipelineStepIndex
import joblib

class PipelineCalculateFov(PipelineStep):
    def __init__(self, pipeline, config):
        super().__init__(pipeline, config)
        self.classifier = joblib.load(self.pipeline.fov_model_path)

    @property
    def index(self) -> PipelineStepIndex:
        return PipelineStepIndex.CalculateFov

    @property
    def required_keys(self) -> list:
        return []

    @property
    def output_keys(self) -> list:
        return ["fov"]

    def run(self, data):
        # Setfov.

        data["fov"] = 60.0

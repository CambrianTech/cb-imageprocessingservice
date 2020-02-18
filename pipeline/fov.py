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
        # Pass latents to sklearn model and store the predicted fov.
        fov_class = self.classifier.predict(data["normals_latents"])[0]
        data["fov"] = 60.0 if fov_class == 0 else 85.0

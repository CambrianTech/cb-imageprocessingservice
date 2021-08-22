import abc
from pipeline.core import PipelineStep

class PipelineInput(PipelineStep, metaclass=abc.ABCMeta):
    def __init__(self, base_path):
        super().__init__()
        self.base_path = base_path

    @property
    def required_keys(self) -> list:
        return ["unique_id"]

    @property
    def output_keys(self) -> list:
        return ["image"]


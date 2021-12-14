import abc
from pipeline.core import PipelineStep, PipelineStepIndex

class PipelineInput(PipelineStep, metaclass=abc.ABCMeta):

    @property
    def index(self) -> PipelineStepIndex:
        return PipelineStepIndex.Input

    @property
    def required_keys(self) -> list:
        return ["unique_id"]

    @property
    def output_keys(self) -> list:
        return ["image"]


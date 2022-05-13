import abc
from abc import abstractmethod
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

    def run():
        raise Exception("call run() method instead")

    @abstractmethod
    def get(self, info: dict) -> dict:
        pass



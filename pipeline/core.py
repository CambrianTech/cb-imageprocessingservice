from abc import ABCMeta, abstractmethod, abstractproperty


class PipelineStep(metaclass=ABCMeta):
    @abstractmethod
    def run(self, data: dict):
        pass

    @abstractproperty
    def required_keys(self) -> list:
        return {}

    @abstractproperty
    def output_keys(self) -> list:
        return {}

    @property
    def config(self) -> dict:
        return self._pipeline.config


class Pipeline:
    def __init__(self, config={}):
        self._steps = []
        self._config = config

    @property
    def config(self):
        return self._config

    def add(self, step: PipelineStep):
        self._steps.append(step)
        step._pipeline = self
        return self

    def validate(self, keys: list):
        for step in self._steps:
            assert all(
                key in keys for key in step.required_keys), "One or more required keys missing for pipeline step %s" % step
            keys += step.output_keys

    def run(self, data: dict) -> dict:
        for step in self._steps:
            assert all(
                key in data for key in step.required_keys), "One or more required keys missing for pipeline step %s" % step
            step.run(data)
            assert all(
                key in data for key in step.output_keys), "One or more output keys missing after pipeline step %s" % step
        return data

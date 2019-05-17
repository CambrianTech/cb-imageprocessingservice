from pipeline.core import PipelineStep


class PipelineUploadResults(PipelineStep):
    @property
    def required_keys(self) -> list:
        return []

    @property
    def output_keys(self) -> list:
        return []

    def run(self, data):
        # TODO
        pass

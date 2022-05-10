import numpy as np
from pipeline.core import PipelineStep, PipelineStepIndex

class PipelineCombinePlaneMasks(PipelineStep):
    @property
    def required_keys(self) -> list:
        return ["room"]

    @property
    def output_keys(self) -> list:
        return ["index_mask"]

    @property
    def index(self) -> PipelineStepIndex:
        return PipelineStepIndex.CombinePlaneMasks

    @property
    def is_batched(self) -> bool:
        return False

    def run_one(self, data):
        room = data["room"]

        index_mask = None

        for i in range(len(room.surfaces)):
            surface = room.surfaces[i]
            mask = surface.final_mask

            if index_mask is None:
                index_mask = np.zeros_like(mask)

            index_mask[mask > 0] = i + 1


        data["index_mask"] = index_mask

    def run(self, data):

        if self.is_batched:
            [self.run_one(datum) for datum in data]
        else:
            self.run_one(data)

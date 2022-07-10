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
        return True

    def run(self, data):
        for datum in data:

            room = datum["room"]

            index_mask = None

            for i in range(len(room.surfaces)):
                surface = room.surfaces[i]
                mask = surface.hires_mask

                if index_mask is None:
                    index_mask = np.zeros_like(mask)

                index_mask[mask > 0] = i + 1


            datum["index_mask"] = index_mask


import numpy as np
from pipeline.core import PipelineStep


def combine_plane_masks(plane_masks: np.ndarray) -> np.ndarray:
    num_planes = len(plane_masks)

    # Start at 1 with plane indices here.
    # Later we subtract 1 so that -1 means no plane, and
    # actual plane indices start at 0.
    plane_indices = np.arange(1, num_planes + 1)

    # Combine the separate plane masks into a single plane mask
    # with the plane indices as values. Also store the alpha
    # values of the masks by summing them.
    if plane_masks.dtype == np.uint8:
        plane_masks = plane_masks.astype(np.float32) / 255

    alpha_mask = np.sum(plane_masks, axis=0, dtype=np.float32)

    bool_masks = plane_masks.astype(np.bool).astype(np.float32)

    # [NumPlanes] @ [NumPlanes, H, W] => [H, W]
    # -1: No plane
    # range(nPlanes): plane index
    index_mask = np.einsum(
        "i,ihw->hw", plane_indices, bool_masks
    ).astype(np.int16) - 1

    return index_mask, alpha_mask


class PipelineCombinePlaneMasks(PipelineStep):
    @property
    def required_keys(self) -> list:
        return ["planes"]

    @property
    def output_keys(self) -> list:
        return ["planes"]

    @property
    def is_batched(self) -> bool:
        return True

    def run(self, data):
        for datum in data:
            index_mask, alpha_mask = combine_plane_masks(datum["planes"]["masks"])

            datum["planes_index_mask"] = index_mask
            datum["planes_alpha_mask"] = alpha_mask

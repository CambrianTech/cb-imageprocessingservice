
import os
import typing
import time

from pipeline.core import schedule_and_wait
from pipeline.fov import PipelineCalculateFov
from pipeline.getdata import PipelineBucketSource, PipelineFileSource
from pipeline.primaryangle import PipelineDeterminePrimaryAngles
from pipeline.runmodels import PipelineRunModels
from pipeline.superpixels import PipelineSuperpixels
from pipeline.refineplanemasks import PipelineRefinePlaneMasks
from pipeline.combineplanemasks import PipelineCombinePlaneMasks
from pipeline.uploadresults import PipelineUploadResults
from pipeline.remote import PipelineRemotePlaneDetector, PipelineRemoteNetworks

class Pipeline():

    def __init__(self, semantic_model_path, fov_model_path, bucket_source=None, bucket_dest=None, plane_source=None, plane_dest=None):


        #setup input:
        if bucket_source is not None:
             self.steps = [
                PipelineBucketSource(bucket_source),
                PipelineRemoteNetworks(plane_source),
                PipelineCalculateFov(fov_model_path),
                PipelineRemotePlaneDetector(plane_dest),
                PipelineRunModels(
                    semantic_path=semantic_model_path,
                    hed_path=os.path.join("hed_model", "HED_pretrained_bsds.npz")
                ),
                PipelineDeterminePrimaryAngles(),
                PipelineSuperpixels(),
                PipelineRefinePlaneMasks(),
                PipelineCombinePlaneMasks(),
                PipelineUploadResults(bucket_dest)
            ]
        else:
            self.steps = [
                PipelineFileSource(),
                PipelineCalculateFov(fov_model_path),
                PipelineRunModels(
                    semantic_path=semantic_model_path,
                    hed_path=os.path.join("hed_model", "HED_pretrained_bsds.npz")
                ),
                PipelineDeterminePrimaryAngles(),
                PipelineSuperpixels(),
                PipelineRefinePlaneMasks(),
                PipelineCombinePlaneMasks()
            ]


    def start(self):
        # Start the processing workers for all steps
        for step in self.steps:
            step.start()

    async def process(self, input_dict: typing.Dict):
        total_start_time = time.time()
        for step in self.steps:
            input_dict = await schedule_and_wait(step.schedule, input_dict)
        print("Planes total pipeline time: %.2fs" %
              (time.time() - total_start_time))
        return input_dict

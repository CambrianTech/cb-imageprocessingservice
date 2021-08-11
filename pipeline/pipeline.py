
import os

from pipeline.fov import PipelineCalculateFov
from pipeline.getdata import PipelineGetData
from pipeline.primaryangle import PipelineDeterminePrimaryAngles
from pipeline.runmodels import PipelineRunModels
from pipeline.superpixels import PipelineSuperpixels
from pipeline.refineplanemasks import PipelineRefinePlaneMasks
from pipeline.combineplanemasks import PipelineCombinePlaneMasks
from pipeline.uploadresults import PipelineUploadResults
from pipeline.remote import PipelineRemotePlaneDetector, PipelineRemoteNetworks

class Pipeline():

    def __init__(self, source, dest, semantic_model_path, fov_model_path, plane_source_url=None, plane_dest_url=None):


        #setup input:
        if plane_source_url is not None:
             self.steps = [
                PipelineGetData(bucket_name=source),
                PipelineRemoteNetworks(plane_source_url),
                PipelineCalculateFov(fov_model_path),
                PipelineRemotePlaneDetector(plane_dest_url),
                PipelineRunModels(
                    semantic_path=semantic_model_path,
                    hed_path=os.path.join("hed_model", "HED_pretrained_bsds.npz")
                ),
                PipelineDeterminePrimaryAngles(),
                PipelineSuperpixels(),
                PipelineRefinePlaneMasks(),
                PipelineCombinePlaneMasks(),
                PipelineUploadResults(bucket_name=dest)
            ]
        else:
            self.steps = [
                PipelineGetData(local_directory=source),
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


    def start():
        # Start the processing workers for all steps
        for step in steps:
            step.start()

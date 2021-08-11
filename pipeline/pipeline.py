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

    def __init__(self, source, destination, semantic_model_path, fov_model_path, plane_url=None):


        #setup input:
        if plane_url is not None:
             self.steps = [
                PipelineGetData(source_bucket, source_directory),
                PipelineRemoteNetworks("http://localhost:%d" % cpu_networks_port),
                PipelineCalculateFov(fov_model_path),
                PipelineRemotePlaneDetector(plane_url),
                PipelineRunModels(
                    semantic_path=semantic_model_path,
                    hed_path=join("hed_model", "HED_pretrained_bsds.npz")
                ),
                PipelineDeterminePrimaryAngles(),
                PipelineSuperpixels(),
                PipelineRefinePlaneMasks(results_bucket),
                PipelineCombinePlaneMasks(),
                PipelineUploadResults(results_bucket)
            ]
        else:
            self.steps = []
            self.steps.append(PipelineGetData(local_directory=source))


        self.steps.extend([
            PipelineRunModels(
                semantic_path=semantic_model_path,
                hed_path=join("hed_model", "HED_pretrained_bsds.npz")
            ),
            PipelineDeterminePrimaryAngles(),
            PipelineSuperpixels(),
            PipelineRefinePlaneMasks(destination),
            PipelineCombinePlaneMasks(),
            PipelineUploadResults(destination)
        ])


    def start():
        # Start the processing workers for all steps
        for step in steps:
            step.start()

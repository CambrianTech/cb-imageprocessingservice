
import os
import typing
import time
import subprocess

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

    def __init__(self, model_path, semantic_model_path, fov_model_path, planes_network_url, bucket_source=None, bucket_dest=None):


        print("Starting CPU networks process")
        cpu_networks_port = 8082
        subprocess.Popen(["python3", "runcpunetworks.py", model_path, str(cpu_networks_port)])

        # Create the steps we want to use in the pipelines
        remote_path = "http://localhost:%d" % cpu_networks_port

        #setup input:
        if bucket_source is not None:
             self.steps = [
                PipelineBucketSource(bucket_source),
                PipelineRemoteNetworks(remote_path),
                PipelineCalculateFov(fov_model_path),
                PipelineRemotePlaneDetector(planes_network_url),
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
                PipelineRemoteNetworks(remote_path),
                PipelineCalculateFov(fov_model_path),
                PipelineRemotePlaneDetector(planes_network_url),
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

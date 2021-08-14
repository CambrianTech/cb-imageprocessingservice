
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

from enum import IntEnum

#labels for ade20k, subtract = 1 for output number. 

class PipelineMode(IntEnum):
    Serve = 0
    Process = 1
    Restore = 2

class PipelineStep(IntEnum):
    RemoteNetworks = 0
    CalculateFov = 1
    RemotePlaneDetector = 2
    RunModels = 3
    DeterminePrimaryAngles = 4
    Superpixels = 5
    RefinePlaneMasks = 6
    CombinePlaneMasks = 7


class Pipeline():

    def __init__(self, mode:PipelineMode, model_path=None, semantic_model_path=None, fov_model_path=None, 
        planes_url=None, bucket_source=None, bucket_dest=None, cpu_networks_port = 8082, restore_step:PipelineStep=None):

        self.model_path = model_path
        self.semantic_model_path = semantic_model_path
        self.fov_model_path = fov_model_path 
        self.planes_url = planes_url 
        self.bucket_source = bucket_source 
        self.bucket_dest = bucket_dest
        self.cpu_networks_port = cpu_networks_port
        self.remote_path = "http://localhost:%d" % cpu_networks_port

        # Create the steps we want to use in the pipelines
        if mode == PipelineMode.Serve:
             self.steps = [
                PipelineBucketSource(self.bucket_source),
                PipelineRemoteNetworks(self.remote_path),
                PipelineCalculateFov(self.fov_model_path),
                PipelineRemotePlaneDetector(self.planes_url),
                PipelineRunModels(semantic_path=self.semantic_model_path, hed_path=os.path.join("hed_model", "HED_pretrained_bsds.npz")),
                PipelineDeterminePrimaryAngles(),
                PipelineSuperpixels(),
                PipelineRefinePlaneMasks(),
                PipelineCombinePlaneMasks(),
                PipelineUploadResults(self.bucket_dest)
            ]

        elif mode == PipelineMode.Process:
            self.steps = [
                PipelineFileSource(),
                PipelineRemoteNetworks(self.remote_path),
                PipelineCalculateFov(self.fov_model_path),
                PipelineRemotePlaneDetector(self.planes_url),
                PipelineRunModels(semantic_path=self.semantic_model_path, hed_path=os.path.join("hed_model", "HED_pretrained_bsds.npz")),
                PipelineDeterminePrimaryAngles(),
                PipelineSuperpixels(),
                PipelineRefinePlaneMasks(),
                PipelineCombinePlaneMasks()
            ]

        elif mode == PipelineMode.Restore:
            print("Restoring from", restore_step.name)

            self.steps = [PipelineFileSource()]

            if restore_step >= PipelineStep.RemoteNetworks:
                self.steps.append(PipelineRemoteNetworks(self.remote_path))
            if restore_step >= PipelineStep.CalculateFov:
                self.steps.append(PipelineCalculateFov(self.fov_model_path))
            if restore_step >= PipelineStep.RemotePlaneDetector:
                self.steps.append(PipelineRemotePlaneDetector(self.planes_url))
            if restore_step >= PipelineStep.RunModels:
                self.steps.append(PipelineRunModels(semantic_path=self.semantic_model_path, hed_path=os.path.join("hed_model", "HED_pretrained_bsds.npz")))
            if restore_step >= PipelineStep.DeterminePrimaryAngles:
                self.steps.append(PipelineDeterminePrimaryAngles())
            if restore_step >= PipelineStep.Superpixels:
                self.steps.append(PipelineSuperpixels())
            if restore_step >= PipelineStep.RefinePlaneMasks: 
                self.steps.append(PipelineRefinePlaneMasks())
            if restore_step >= PipelineStep.CombinePlaneMasks:
                self.steps.append(PipelineCombinePlaneMasks())
            


    def start(self):
        print("Starting")
        
        subprocess.Popen(["python3", "runcpunetworks.py", self.model_path, str(self.cpu_networks_port)])

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

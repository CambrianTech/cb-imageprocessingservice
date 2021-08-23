
import os
import time
import subprocess

from pipeline.core import schedule_and_wait, PipelineStep
from pipeline.logging import get_unique_id, set_logging_dir, set_logging_step, log_data, LogLevel, set_logging_level

from pipeline.s3input import PipelineS3Input
from pipeline.fileinput import PipelineFileInput
from pipeline.s3output import PipelineS3Output
from pipeline.fileoutput import PipelineFileOutput

from pipeline.fov import PipelineCalculateFov
from pipeline.primaryangle import PipelineDeterminePrimaryAngles
from pipeline.runmodels import PipelineRunModels
from pipeline.superpixels import PipelineSuperpixels
from pipeline.refineplanemasks import PipelineRefinePlaneMasks
from pipeline.combineplanemasks import PipelineCombinePlaneMasks
from pipeline.remote import PipelineRemotePlaneDetector, PipelineRemoteNetworks

from enum import IntEnum

#labels for ade20k, subtract = 1 for output number. 

class PipelineMode(IntEnum):
    Serve = 0
    Process = 1
    Restore = 2

class PipelineStepIndex(IntEnum):
    Input = 0
    RemoteNetworks = 1
    CalculateFov = 2
    RemotePlaneDetector = 3
    RunModels = 4
    DeterminePrimaryAngles = 5
    Superpixels = 6
    RefinePlaneMasks = 7
    CombinePlaneMasks = 8
    Output = 9


class Pipeline():

    def __init__(self,  mode:PipelineMode, \
                        model_path=None, semantic_model_path=None, fov_model_path=None, \
                        planes_url=None, src_path=None, dest_path=None, cpu_networks_port = 8082, \
                        restore_step:PipelineStepIndex=None, export_step:PipelineStepIndex=None, \
                        logging_dir=None, logging_level=LogLevel.Nothing, logging_step:PipelineStepIndex=None):

        self.mode = mode

        self.model_path = model_path
        self.semantic_model_path = semantic_model_path
        self.fov_model_path = fov_model_path 
        self.planes_url = planes_url 
        self.src_path = src_path 
        self.dest_path = dest_path
        self.cpu_networks_port = cpu_networks_port
        self.remote_path = "http://localhost:%d" % cpu_networks_port
        self.restore_step = restore_step
        self.export_step = export_step

        self.logging_dir = logging_dir
        self.logging_level = logging_level
        self.logging_step = logging_step

        self.start_step = self.restore_step if self.restore_step is not None else PipelineStepIndex.Input

        # Create the steps we want to use in the pipelines
        if self.mode == PipelineMode.Serve:
            s3Client = S3Client()
            self.steps = [
                PipelineS3Input(self.src_path, s3Client),
                PipelineRemoteNetworks(self.remote_path),
                PipelineCalculateFov(self.fov_model_path),
                PipelineRemotePlaneDetector(self.planes_url),
                PipelineRunModels(semantic_path=self.semantic_model_path, hed_path=os.path.join("hed_model", "HED_pretrained_bsds.npz")),
                PipelineDeterminePrimaryAngles(),
                PipelineSuperpixels(),
                PipelineRefinePlaneMasks(),
                PipelineCombinePlaneMasks(),
                PipelineS3Output(self.dest_path, s3Client)
            ]

        elif self.mode == PipelineMode.Process:
            s3Client = S3Client()
            self.steps = [
                PipelineFileInput(self.src_path, s3Client),
                PipelineRemoteNetworks(self.remote_path),
                PipelineCalculateFov(self.fov_model_path),
                PipelineRemotePlaneDetector(self.planes_url),
                PipelineRunModels(semantic_path=self.semantic_model_path, hed_path=os.path.join("hed_model", "HED_pretrained_bsds.npz")),
                PipelineDeterminePrimaryAngles(),
                PipelineSuperpixels(),
                PipelineRefinePlaneMasks(),
                PipelineCombinePlaneMasks(),
                PipelineFileOutput(self.dest_path, s3Client)
            ]

        elif self.mode == PipelineMode.Restore:
            print("Restoring from", restore_step.name)

            self.steps = [PipelineFileInput(self.src_path)]

            if restore_step <= PipelineStepIndex.RemoteNetworks:
                self.push(PipelineRemoteNetworks(self.remote_path))
            if restore_step <= PipelineStepIndex.CalculateFov:
                self.push(PipelineCalculateFov(self.fov_model_path))
            if restore_step <= PipelineStepIndex.RemotePlaneDetector:
                self.push(PipelineRemotePlaneDetector(self.planes_url))
            if restore_step <= PipelineStepIndex.RunModels:
                self.push(PipelineRunModels(semantic_path=self.semantic_model_path, hed_path=os.path.join("hed_model", "HED_pretrained_bsds.npz")))
            if restore_step <= PipelineStepIndex.DeterminePrimaryAngles:
                self.push(PipelineDeterminePrimaryAngles())
            if restore_step <= PipelineStepIndex.Superpixels:
                self.push(PipelineSuperpixels())
            if restore_step <= PipelineStepIndex.RefinePlaneMasks: 
                self.push(PipelineRefinePlaneMasks())
            if restore_step <= PipelineStepIndex.CombinePlaneMasks:
                self.push(PipelineCombinePlaneMasks())

            self.push(PipelineFileOutput(self.dest_path))


    def start(self):
        print("Starting threads")

        if self.logging_step is not None:
            print("Logging is enabled for step", self.logging_step.name)

        #start RemoteNetworks if needed downstream
        if self.mode != PipelineMode.Restore or self.restore_step <= PipelineStepIndex.RemoteNetworks:
            subprocess.Popen(["python3", "runcpunetworks.py", self.model_path, str(self.cpu_networks_port)])

        # Start the processing workers for all steps
        for step in self.steps:
            step.start()

    def step_index(self, pos:int):
        return PipelineStepIndex(self.start_step + pos - 1)

    def push(self, step:PipelineStep):
        index = self.step_index(len(self.steps))
        print("Appending step", index.name)
        self.steps.append(step)

    async def process(self, data):

        if self.logging_dir is not None and not os.path.exists(self.logging_dir):
            os.makedirs(self.logging_dir)

        total_start_time = time.time()
        index = 0

        for step in self.steps:

            #consider perhaps passing logging down into steps, trigger off that
            logging_dir = None if self.logging_dir is None else "%s/%s" % (self.logging_dir, get_unique_id(data))

            set_logging_dir(data, logging_dir)
            set_logging_level(data, self.logging_level)

            current_step = self.step_index(index)
            
            set_logging_step(data, self.logging_step, current_step)
            print("Step %s" % (current_step.name))

            step_start = time.time()
            data = await schedule_and_wait(step.schedule, data)
            print("Step %s took %.2f seconds" % (current_step.name, time.time() - step_start))

            if current_step == self.export_step and logging_dir is not None:
                log_data(data)

            index += 1

        print("Planes total pipeline time: %.2fs" %
              (time.time() - total_start_time))
        return data

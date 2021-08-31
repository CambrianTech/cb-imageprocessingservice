
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
from pipeline.planegeometry import PipelinePlaneGeometry
from pipeline.refineplanemasks import PipelineRefinePlaneMasks
from pipeline.linefinder import PipelineLineFinder
from pipeline.refine import PipelineRefineResults
from pipeline.combineplanemasks import PipelineCombinePlaneMasks
from pipeline.remote import PipelineRemotePlaneDetector, PipelineRemoteNetworks
from pipeline.poseestimator import PipelinePoseEstimator
from pipeline.extractsurfaces import PipelineExtractSurfaces
from pipeline.surfacerefinement import PipelineSurfaceRefinement
from pipeline.mergesurfaces import PipelineMergeSurfaces


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
    ExtractSurfaces = 6
    Superpixels = 7
    FindLines = 8
    RefineSurfaces = 9
    Geometry = 10
    EstimatePose = 11
    MergeSurfaces = 12
    Refine = 13
    CombinePlaneMasks = 14
    Output = 15

class PipelineNoOp(PipelineStep):

    def __init__(self):
        super().__init__()

    @property
    def required_keys(self) -> list:
        return []

    @property
    def output_keys(self) -> list:
        return []

    def run(self, data):
        print("No Operation")

class Pipeline():

    def __init__(self,  mode:PipelineMode, api_level, \
                        model_path=None, semantic_model_path=None, fov_model_path=None, \
                        planes_url=None, src_path=None, dest_path=None, cpu_networks_port = 8082, \
                        restore_step:PipelineStepIndex=None, export_step:PipelineStepIndex=None, \
                        logging_dir=None, logging_level=LogLevel.Nothing, logging_step:PipelineStepIndex=None):

        self.mode = mode
        self.api_level = api_level
        self._running = False

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

        #todo: consolidate steps around start_step and eliminate three if statements below
        if self.mode == PipelineMode.Serve:
            s3Client = S3Client()
            input_step = PipelineS3Input(self.src_path, s3Client)
            output_step = PipelineS3Output(self.dest_path, s3Client, api_level=self.api_level)
        else:
            input_step = PipelineFileInput(self.src_path)
            output_step = PipelineFileOutput(self.dest_path, api_level=self.api_level)

        superpixels_step = PipelineSuperpixels if self.api_level < 3 else PipelineNoOp

        extract_step = PipelineExtractSurfaces

        if self.api_level < 3.5:
            refine_surfaces_step = PipelineNoOp
            lines_step = PipelineNoOp
            geometry_step = PipelineNoOp
            estimate_pose_step = PipelineNoOp
            merge_step = PipelineNoOp
            refine_step = PipelineRefinePlaneMasks
        else:
            refine_surfaces_step = PipelineSurfaceRefinement
            lines_step = PipelineLineFinder
            geometry_step = PipelinePlaneGeometry
            estimate_pose_step = PipelinePoseEstimator
            merge_step = PipelineMergeSurfaces
            refine_step = PipelineRefineResults

        # Create the steps we want to use in the pipelines
        if self.mode == PipelineMode.Serve:
            
            self.steps = [
                input_step,
                PipelineRemoteNetworks(self.remote_path),
                PipelineCalculateFov(self.fov_model_path),
                PipelineRemotePlaneDetector(self.planes_url),
                PipelineRunModels(semantic_path=self.semantic_model_path, hed_path=os.path.join("hed_model", "HED_pretrained_bsds.npz")),
                PipelineDeterminePrimaryAngles(),
                extract_step(),
                refine_surfaces_step(),
                superpixels_step(),
                lines_step(),
                geometry_step(),
                estimate_pose_step(),
                merge_step(),
                refine_step(),
                PipelineCombinePlaneMasks(),
                output_step
            ]

        elif self.mode == PipelineMode.Process:
            self.steps = [
                input_step,
                PipelineRemoteNetworks(self.remote_path),
                PipelineCalculateFov(self.fov_model_path),
                PipelineRemotePlaneDetector(self.planes_url),
                PipelineRunModels(semantic_path=self.semantic_model_path, hed_path=os.path.join("hed_model", "HED_pretrained_bsds.npz")),
                PipelineDeterminePrimaryAngles(),
                extract_step(),
                refine_surfaces_step(),
                superpixels_step(),
                lines_step(),
                geometry_step(),
                estimate_pose_step(),
                merge_step(),
                refine_step(),
                PipelineCombinePlaneMasks(),
                output_step
            ]

        elif self.mode == PipelineMode.Restore:
            print("Restoring from", restore_step.name)

            self.steps = [input_step]

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
            if restore_step <= PipelineStepIndex.ExtractSurfaces:
                self.push(extract_step())
            if restore_step <= PipelineStepIndex.Superpixels:
                self.push(superpixels_step())
            if restore_step <= PipelineStepIndex.FindLines: 
                self.push(lines_step())
            if restore_step <= PipelineStepIndex.RefineSurfaces:
                self.push(refine_surfaces_step())
            if restore_step <= PipelineStepIndex.Geometry: 
                self.push(geometry_step())
            if restore_step <= PipelineStepIndex.EstimatePose: 
                self.push(estimate_pose_step())
            if restore_step <= PipelineStepIndex.MergeSurfaces: 
                self.push(merge_step())
            if restore_step <= PipelineStepIndex.Refine: 
                self.push(refine_step())
            if restore_step <= PipelineStepIndex.CombinePlaneMasks:
                self.push(PipelineCombinePlaneMasks())

            self.push(output_step)


    def start(self):
        print("Starting threads")
        self._running = True

        if self.logging_step is not None:
            print("Logging is enabled for step", self.logging_step.name)

        #start RemoteNetworks if needed downstream
        if self.mode != PipelineMode.Restore or self.restore_step <= PipelineStepIndex.RemoteNetworks:
            subprocess.Popen(["python3", "runcpunetworks.py", self.model_path, str(self.cpu_networks_port)])

        # Start the processing workers for all steps
        for step in self.steps:
            step.start()

    def stop(self):
        print("Stopping threads")
        self._running = False
        for step in self.steps:
            step.stop()

    @property
    def running(self):
        return self._running

    def step_index(self, pos:int):
        return PipelineStepIndex(self.start_step + pos - 1)

    def push(self, step:PipelineStep):
        index = self.step_index(len(self.steps))
        print("Appending step", index.name)
        self.steps.append(step)

    async def process(self, data):

        if (len(self.steps) == 0): return

        if self.logging_dir is not None and not os.path.exists(self.logging_dir):
            os.makedirs(self.logging_dir)

        total_start_time = time.time()
        index = 0

        first_step = self.step_index(0)
        last_step = self.step_index(len(self.steps)-1)

        print("\n##### Processing %d steps: %s(%d) - %s(%d) #####" % (len(self.steps), first_step.name, int(first_step), last_step.name, int(last_step)))

        for step in self.steps:

            if not self.running: break

            #consider perhaps passing logging down into steps, trigger off that
            logging_dir = None if self.logging_dir is None else "%s/%s" % (self.logging_dir, get_unique_id(data))

            set_logging_dir(data, logging_dir)
            set_logging_level(data, self.logging_level)

            current_step = self.step_index(index)
            
            set_logging_step(data, self.logging_step, current_step)
            print("%d) %s" % (int(current_step), current_step.name))

            step_start = time.time()
            data = await schedule_and_wait(step.schedule, data)
            print("%s(%d) took %.2f seconds" % (current_step.name, int(current_step), time.time() - step_start))

            if current_step == self.export_step and logging_dir is not None:
                log_data(data)

            index += 1

        print("##### Planes total pipeline time: %.2f seconds #####\n" %
              (time.time() - total_start_time))
        return data


import os
import time
import subprocess
from enum import IntEnum

from pipeline.core import schedule_and_wait, PipelineStep, PipelineStepIndex
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

class PipelineNoOp(PipelineStep):

    def __init__(self, pipeline, index):
        super().__init__(pipeline)
        self._index = PipelineStepIndex(index)

    @property
    def index(self) -> PipelineStepIndex:
        return self._index

    @property
    def required_keys(self) -> list:
        return []

    @property
    def output_keys(self) -> list:
        return []

    def run(self, data):
        print("Passthrough")

    @property
    def description(self) -> str:
        return "%d) %s (noop)" % (int(self.index), self.index.name) 

class PipelineMode(IntEnum):
    Serve = 0
    Process = 1
    Restore = 2

#labels for ade20k, subtract = 1 for output number. 
class Pipeline():

    def __init__(self,  mode:PipelineMode, api_level, \
                        model_path=None, semantic_model_path=None, fov_model_path=None, hed_model_path=None, \
                        planes_url=None, src_path=None, dest_path=None, cpu_networks_port = 8082, \
                        restore_step:PipelineStepIndex=None, export_step:PipelineStepIndex=None, stop_step=None, \
                        logging_dir=None, logging_level=LogLevel.Nothing, logging_step:PipelineStepIndex=None):

        self.mode = PipelineMode(mode)
        self.api_level = api_level
        self._running = False

        self.model_path = model_path
        self.semantic_model_path = semantic_model_path
        self.hed_model_path = hed_model_path

        self.fov_model_path = fov_model_path 
        self.planes_url = planes_url 
        self.src_path = src_path 
        self.dest_path = dest_path
        self.cpu_networks_port = cpu_networks_port
        self.remote_path = "http://localhost:%d" % cpu_networks_port
        self.restore_step = None if restore_step is None else PipelineStepIndex(restore_step)
        self.export_step = None if export_step is None else PipelineStepIndex(export_step)

        self.logging_dir = logging_dir
        self.logging_level = logging_level
        self.logging_step = None if logging_step is None else PipelineStepIndex(logging_step)

        self.start_step = PipelineStepIndex(self.restore_step if self.restore_step is not None else PipelineStepIndex.Input + 1)

        if stop_step is None:
            self.stop_step = PipelineStepIndex(self.export_step if self.export_step is not None else PipelineStepIndex.Output)
        else:
            self.stop_step = PipelineStepIndex(stop_step)

        self.assemble()

    def assemble(self):

        #Important: some devices may not be able to instantiate a class, so a list is first built
        if self.mode == PipelineMode.Serve:
            self.s3_client = S3Client()
            input_step = PipelineS3Input
            output_step = PipelineS3Output
        else:
            input_step = PipelineFileInput
            output_step = PipelineFileOutput

        all_steps = list([None] * (PipelineStepIndex.Output + 1))

        all_steps[PipelineStepIndex.RemoteNetworks] = PipelineRemoteNetworks
        all_steps[PipelineStepIndex.CalculateFov] = PipelineCalculateFov
        all_steps[PipelineStepIndex.RemotePlaneDetector] = PipelineRemotePlaneDetector
        all_steps[PipelineStepIndex.RunModels] = PipelineRunModels
        all_steps[PipelineStepIndex.DeterminePrimaryAngles] = PipelineDeterminePrimaryAngles
        all_steps[PipelineStepIndex.ExtractSurfaces] = PipelineExtractSurfaces
        all_steps[PipelineStepIndex.FindLines] = PipelineLineFinder
        all_steps[PipelineStepIndex.Geometry] = PipelinePlaneGeometry
        all_steps[PipelineStepIndex.EstimatePose] = PipelinePoseEstimator
        all_steps[PipelineStepIndex.RefineSurfaces] = PipelineSurfaceRefinement
        all_steps[PipelineStepIndex.MergeSurfaces] = PipelineMergeSurfaces
        all_steps[PipelineStepIndex.Refine] = PipelineRefineResults
        all_steps[PipelineStepIndex.Superpixels] = PipelineSuperpixels if self.api_level < 3 else None
        all_steps[PipelineStepIndex.CombinePlaneMasks] = PipelineCombinePlaneMasks
        all_steps[PipelineStepIndex.Output] = output_step
        
        if self.api_level < 3.5:
            all_steps[PipelineStepIndex.RefineSurfaces] = None
            all_steps[PipelineStepIndex.FindLines] = None
            all_steps[PipelineStepIndex.Geometry] = None
            all_steps[PipelineStepIndex.EstimatePose] = None
            all_steps[PipelineStepIndex.MergeSurfaces] = None
            all_steps[PipelineStepIndex.Refine] = PipelineRefinePlaneMasks
        

        self.steps = []
        self.push(input_step(self))

        for index in range(self.start_step, self.stop_step + 1):

            initializer = all_steps[index]
            if not initializer is None:
                self.push(initializer(self))            


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

    def push(self, step:PipelineStep):
        print("Appending step %s" % step.description)
        self.steps.append(step)

    async def process(self, data):

        if (len(self.steps) == 0): return

        if self.logging_dir is not None and not os.path.exists(self.logging_dir):
            os.makedirs(self.logging_dir)

        start_time = time.time()

        print("\n##### Running stages %s through %s #####" % (self.steps[1].description, self.steps[len(self.steps)-1].description))

        for step in self.steps:

            if not self.running: break

            #consider perhaps passing logging down into steps, trigger off that
            logging_dir = None if self.logging_dir is None else os.path.join(self.logging_dir, get_unique_id(data))

            set_logging_dir(data, logging_dir)
            set_logging_level(data, self.logging_level)
            
            set_logging_step(data, self.logging_step, step.index)
            print(step.description)

            step_start = time.time()
            data = await schedule_and_wait(step.schedule, data)
            print("%s took %.2f seconds" % (step.description, time.time() - step_start))

            if step.index == self.export_step and logging_dir is not None:
                log_data(data)

        print("##### All stages time: %.2f seconds #####\n" % (time.time() - start_time))

        return data

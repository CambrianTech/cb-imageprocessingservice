import os
import time
from enum import IntEnum
from termcolor import colored

from .config import PipelineMode, PipelineConfig
from .core import schedule_and_wait, PipelineStep, PipelineStepIndex
from .data.logging import get_unique_id, set_logging_dir, set_logging_step, log_data, set_logging_level

from .stages.aws.s3client import S3Client
from .stages.fileinput import PipelineFileInput
from .stages.fileoutput import PipelineFileOutput

from .stages.fov import PipelineCalculateFov
from .stages.primaryangle import PipelineDeterminePrimaryAngles
from .stages.segmentation import PipelineSemanticSegmentation
from .stages.edgedetector import PipelineEdgeDetector
from .stages.superpixels import PipelineSuperpixels
from .stages.planegeometry import PipelinePlaneGeometry
from .stages.refineplanemasks import PipelineRefinePlaneMasks
from .stages.linefinder import PipelineLineFinder
from .stages.combineplanemasks import PipelineCombinePlaneMasks
from .stages.planedetector import PipelinePlaneDetector
from .stages.reverserenderer import PipelineReverseRenderer
from .stages.poseestimator import PipelinePoseEstimator
from .stages.extractsurfaces import PipelineExtractSurfaces
from .stages.surfacerefinement import PipelineSurfaceRefinement
from .stages.roomsolver import PipelineRoomSolver
from .stages.vanishingpointfinder import PipelineVanishingPointFinder
from .stages.barrierfinder import PipelineBarrierFinder
from .stages.trimfinder import PipelineTrimFinder
from .stages.legfinder import PipelineLegFinder
from .stages.aws.s3input import PipelineS3Input
from .stages.aws.s3output import PipelineS3Output

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

#labels for ade20k, subtract = 1 for output number. 
class Pipeline():

    def __init__(self, config):

        self.config = config
        self._running = False

        self.start_step = PipelineStepIndex(self.config.restore_step if self.config.restore_step is not None else PipelineStepIndex.Input + 1)

        if self.config.stop_step is None:
            self.stop_step = PipelineStepIndex(self.config.export_step if self.config.export_step is not None else PipelineStepIndex.Output)
        else:
            self.stop_step = PipelineStepIndex(self.config.stop_step)

        self.assemble()

    def assemble(self):

        #Important: some devices may not be able to instantiate a class, so a list is first built
        if self.config.mode == PipelineMode.Serve:
            self.s3_client = S3Client()
            input_step = PipelineS3Input
            output_step = PipelineS3Output
        else:
            input_step = PipelineFileInput
            output_step = PipelineFileOutput

        all_steps = list([None] * (PipelineStepIndex.Output + 1))

        all_steps[PipelineStepIndex.CalculateFov] = PipelineCalculateFov
        all_steps[PipelineStepIndex.PlaneDetector] = PipelinePlaneDetector
        all_steps[PipelineStepIndex.Segmentation] = PipelineSemanticSegmentation
        all_steps[PipelineStepIndex.EdgeDetector] = PipelineEdgeDetector
        all_steps[PipelineStepIndex.ReverseRenderer] = PipelineReverseRenderer
        all_steps[PipelineStepIndex.DeterminePrimaryAngles] = PipelineDeterminePrimaryAngles
        all_steps[PipelineStepIndex.ExtractSurfaces] = PipelineExtractSurfaces
        all_steps[PipelineStepIndex.FindLines] = PipelineLineFinder
        all_steps[PipelineStepIndex.SolveRoom] = PipelineRoomSolver
        all_steps[PipelineStepIndex.VanishingPoints] = None
        all_steps[PipelineStepIndex.Barriers] = None
        all_steps[PipelineStepIndex.FindTrim] = None
        all_steps[PipelineStepIndex.FindLegs] = None
        all_steps[PipelineStepIndex.Geometry] = PipelinePlaneGeometry
        all_steps[PipelineStepIndex.EstimatePose] = PipelinePoseEstimator
        all_steps[PipelineStepIndex.Refine] = PipelineSurfaceRefinement
        all_steps[PipelineStepIndex.Superpixels] = None
        all_steps[PipelineStepIndex.CombinePlaneMasks] = None
        all_steps[PipelineStepIndex.Output] = output_step
        

        #todo: deprecate these:
        #all_steps[PipelineStepIndex.EstimatePose] = None
        #all_steps[PipelineStepIndex.Geometry] = None
        
        print("Initializing steps %d through %d" % (self.start_step, self.stop_step))

        self.steps = []
        self.push(input_step(self))

        for index in range(self.start_step, self.stop_step + 1):

            initializer = all_steps[index]
            if not initializer is None:
                print("Initializing step", PipelineStepIndex(index))
                self.push(initializer(self))
                print(PipelineStepIndex(index), "Added")    


    def start(self):
        print("Starting threads")
        self._running = True

        if self.config.logging_step is not None:
            print("Logging is enabled for step", self.config.logging_step.name)

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

        if self.config.logging_dir is not None and not os.path.exists(self.config.logging_dir):
            os.makedirs(self.config.logging_dir)

        start_time = time.time()

        print(colored("Running stages %s through %s" % (self.steps[1].description, self.steps[len(self.steps)-1].description), attrs=['bold']))

        for step in self.steps:

            if not self.running: break

            #consider perhaps passing logging down into steps, trigger off that
            logging_dir = None if self.config.logging_dir is None else os.path.join(self.config.logging_dir, data["unique_id"])

            set_logging_dir(data, logging_dir)
            set_logging_level(data, self.config.logging_level)
            
            set_logging_step(data, self.config.logging_step, step.index)
            print(step.description)

            step_start = time.time()
            data = await schedule_and_wait(step.schedule, data)
            print("%s took %.2f seconds" % (step.description, time.time() - step_start))

            if step.index == self.config.export_step and logging_dir is not None:
                log_data(data)

        print(colored("All stages time: %.2f seconds\n" % (time.time() - start_time), attrs=['bold']))

        return data

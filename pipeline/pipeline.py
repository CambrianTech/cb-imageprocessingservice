import os
import time
from enum import IntEnum
from termcolor import colored
import asyncio
import typing
import psutil

from .config import PipelineMode, PipelineConfig
from .core import PipelineStep, PipelineStepIndex
from .data.logging import get_unique_id, set_logging_dir, set_logging_step, log_data, set_logging_level
from .misc.utils import get_memory_usage_mb

from .stages.aws.s3client import S3Client
from .stages.fileinput import PipelineFileInput
from .stages.fileoutput import PipelineFileOutput

from .stages.fov import PipelineCalculateFov
from .stages.primaryangle import PipelineDeterminePrimaryAngles
from .stages.segmentation import PipelineSemanticSegmentation
from .stages.edgedetector import PipelineEdgeDetector
from .stages.superpixels import PipelineSuperpixels
from .stages.planegeometry import PipelinePlaneGeometry
from .stages.linefinder import PipelineLineFinder
from .stages.combineplanemasks import PipelineCombinePlaneMasks
from .stages.planedetector import PipelinePlaneDetector
from .stages.reverserenderer import PipelineReverseRenderer
from .stages.poseestimator import PipelinePoseEstimator
from .stages.extractsurfaces import PipelineExtractSurfaces
from .stages.surfacerefinement import PipelineSurfaceRefinement
from .stages.scenegenerator import PipelineSceneGenerator
from .stages.surfacesolver import PipelineSurfaceSolver
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
            self.input_step = PipelineS3Input(self)
            output_step = PipelineS3Output
        else:
            self.input_step = PipelineFileInput(self)
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
        all_steps[PipelineStepIndex.GenerateScene] = PipelineSceneGenerator
        all_steps[PipelineStepIndex.SolveSurfaces] = PipelineSurfaceSolver
        all_steps[PipelineStepIndex.VanishingPoints] = PipelineVanishingPointFinder
        all_steps[PipelineStepIndex.Barriers] = None
        all_steps[PipelineStepIndex.FindTrim] = None
        all_steps[PipelineStepIndex.FindLegs] = None
        all_steps[PipelineStepIndex.Geometry] = PipelinePlaneGeometry
        all_steps[PipelineStepIndex.EstimatePose] = PipelinePoseEstimator
        all_steps[PipelineStepIndex.Refine] = PipelineSurfaceRefinement
        all_steps[PipelineStepIndex.Superpixels] = None
        all_steps[PipelineStepIndex.CombinePlaneMasks] = PipelineCombinePlaneMasks
        all_steps[PipelineStepIndex.Output] = output_step
        

        #todo: deprecate these:
        #all_steps[PipelineStepIndex.EstimatePose] = None
        #all_steps[PipelineStepIndex.Geometry] = None
        
        print("Initializing steps %d through %d" % (self.start_step, self.stop_step))

        self.steps = list()

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

    async def stop(self):
        if not self._running: return

        print("Stopping threads")
        self._running = False

        for step in self.steps:
            await step.stop()

        print("Stopped all threads")

    def kill(self):

        for task in asyncio.all_tasks():
            task.cancel()

        asyncio.get_event_loop().stop()

    @property
    def running(self):
        return self._running

    def push(self, step:PipelineStep):
        print("Appending step %s" % step.description)
        self.steps.append(step)

    async def process(self, data, step_callback=None):

        if self.config.logging_dir is not None and not os.path.exists(self.config.logging_dir):
            os.makedirs(self.config.logging_dir)

        def log_step(step, start):
            print("%s took %.2f seconds" % (step.description, time.time() - start))
            print(colored("Current memory at %.2f MB" % get_memory_usage_mb(), attrs=['bold']))

            if step.index == self.config.export_step and logging_dir is not None:
                log_data(data)

        print(colored("Running steps %d through %d" % (self.start_step, self.stop_step), attrs=['bold']))

        start_time = time.time()

        data = self.input_step.run(data)

        if "unique_id" not in data:
            if 'image_s3_key' in data:
                data['unique_id'] = data['image_s3_key']
                print("Warning 'image_s3_key' is no longer being used. Please update this to 'unique_id'")
            else:
                print("Invalid data, unique_id not provided")
                return None

        logging_dir = None if self.config.logging_dir is None else os.path.join(self.config.logging_dir, data["unique_id"])

        set_logging_dir(data, logging_dir)
        set_logging_level(data, self.config.logging_level)

        log_step(self.input_step, start_time)

        for step in self.steps:

            if not self.running: break
            
            set_logging_step(data, self.config.logging_step, step.index)
            print(step.description)

            step_start = time.time()

            await schedule_and_wait(step.schedule, data)

            if step_callback is not None: 
                step_callback()

            log_step(step, step_start)

        print(colored("All stages time: %.2f seconds\n" % (time.time() - start_time), attrs=['bold']))

        #all memory must be cleaned up by this point (data elements may reference each other):
        for key in data:
            data[key] = None

def schedule_and_wait(func: typing.Callable[[typing.Dict, asyncio.Future], None], input_dict: typing.Dict) -> asyncio.Future:
    """Calls a function and returns a future that the function is supposed to fullfil."""
    future = asyncio.get_event_loop().create_future()
    func(input_dict, future)
    return future

async def merge_future_dicts(*futures) -> typing.Dict:
    """Merges dict outputs of futures into a single dict."""
    merged_dict = {}
    for result in asyncio.as_completed(futures):
        merged_dict.update(result)
    return merged_dict

def num_waiting_items(steps: typing.List[PipelineStep]) -> int:
    """Counts the number of waiting items in a list of pipeline steps."""
    return sum([step.num_waiting_items for step in steps])

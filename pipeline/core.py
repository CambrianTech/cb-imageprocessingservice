from abc import ABCMeta, abstractmethod
from time import time
import asyncio
import typing
from multiprocessing import cpu_count
from enum import IntEnum, auto
from termcolor import colored
from types import SimpleNamespace
import traceback

from .config import PipelineMode, PipelineConfig

class PipelineStepIndex(IntEnum):
    Input = 0
    CalculateFov = auto()
    PlaneDetector = auto()
    EdgeDetector = auto()
    Segmentation = auto()
    ReverseRenderer = auto()
    DeterminePrimaryAngles = auto()
    ExtractSurfaces = auto()
    FindLines = auto()
    Geometry = auto()
    GenerateScene = auto()
    VanishingPoints = auto()
    SolveSurfaces = auto()
    FindTrim = auto()
    FindLegs = auto()
    EstimatePose = auto()
    Refine = auto()
    Superpixels = auto()
    CombinePlaneMasks = auto()
    Output = auto()

class PipelineStep(metaclass=ABCMeta):
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.config = pipeline.config
        self._running = False
        self._input_queue = asyncio.Queue()

    @abstractmethod
    def run(self, data: dict):
        pass

    @property
    @abstractmethod
    def required_keys(self) -> list:
        return {}

    @property
    @abstractmethod
    def output_keys(self) -> list:
        return {}

    @property
    @abstractmethod
    def index(self) -> PipelineStepIndex:
        return None

    @property
    def description(self) -> str:
        return "%d) %s" % (int(self.index), self.index.name) if self.index is not None else "PipelineStep"

    @property
    def is_batched(self) -> bool:
        return False

    @property
    def started(self) -> bool:
        return self._running

    @property
    def num_waiting_items(self) -> int:
        return self._input_queue.qsize()

    def schedule(self, input_dict, future):
        self.future = future
        self._input_queue.put_nowait((input_dict, future))

    def start(self):
        if not self._running:
            self._running = True
            for _ in range(1 if self.is_batched else cpu_count()):
                asyncio.ensure_future(self.run_step_in_background())

    async def stop(self):
        print("Stopping %s" % self.description)
        self._running = False
        await self._input_queue.join()
        print("%s Stopped" % self.description)

    async def run_step_in_background(self):

        while self._running:

            datum, result_future = await self._input_queue.get()
            self._input_queue.task_done()

            if result_future.cancelled():
                continue

            result_futures = [result_future]
            data = [datum]

            if self.is_batched:
                debounce_start_time = time()
                wait_time = 0

                # Add items from the queue while more are available.
                # Do this by waiting a small amount of time for new data
                # up to a maximum time.
                while wait_time < self.config.batch_max_wait_time and len(data) < self.config.batch_max_size:
                    await asyncio.sleep(self.config.batch_debounce_time)
                    try:
                        # Dequeue until we have enough for a batch
                        # or until we run out.
                        while len(data) < self.config.batch_max_size:
                            datum, result_future = self._input_queue.get_nowait()
                            if not result_future.cancelled():
                                result_futures.append(result_future)
                                data.append(datum)
                    except asyncio.QueueEmpty:
                        pass
                    wait_time = time() - debounce_start_time

            # Run the data
            step_start_time = time()
            try:
                await asyncio.get_event_loop().run_in_executor(None, self.run, data[0] if not self.is_batched else data)
            except Exception as e:
                traceback.print_exc()
                
                for result_future in result_futures:
                    if not result_future.cancelled():
                        result_future.set_exception(e)
                continue

            for result_future, datum in zip(result_futures, data):
                if not result_future.cancelled():
                    result_future.set_result(datum)
                        

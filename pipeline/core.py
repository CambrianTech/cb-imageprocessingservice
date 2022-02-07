from abc import ABCMeta, abstractmethod
from time import time
import asyncio
import typing
from multiprocessing import cpu_count
from enum import IntEnum
from termcolor import colored
from types import SimpleNamespace

class PipelineStepIndex(IntEnum):
    Input = 0
    RemoteNetworks = 1
    CalculateFov = 2
    RemotePlaneDetector = 3
    RunModels = 4
    DeterminePrimaryAngles = 5
    ExtractSurfaces = 6
    FindLines = 7
    Geometry = 8
    SolveRoom = 9
    VanishingPoints = 10
    Barriers = 11
    FindTrim = 12
    FindLegs = 13
    EstimatePose = 14
    Refine = 15
    Superpixels = 16
    CombinePlaneMasks = 17
    Output = 18

class PipelineStepConfig(SimpleNamespace):
    use_gpu=True
    batch_max_wait_time=1.0
    batch_debounce_time=0.2
    batch_max_size=1

class PipelineStep(metaclass=ABCMeta):
    def __init__(self, pipeline, config:PipelineStepConfig):
        self.pipeline = pipeline
        self.config = config
        self.batch_max_wait_time = config.batch_max_wait_time
        self.batch_debounce_time = config.batch_debounce_time
        self.batch_max_size = config.batch_max_size
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

    def stop(self):
        self._running = False

    async def run_step_in_background(self):
        loop = asyncio.get_event_loop()

        while self._running:
            #sleep on running and entries
            if self._input_queue.empty():
                await asyncio.sleep(0.0001)
                continue

            datum, result_future = self._input_queue.get_nowait()
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
                while wait_time < self.batch_max_wait_time and len(data) < self.batch_max_size:
                    await asyncio.sleep(self.batch_debounce_time)
                    try:
                        # Dequeue until we have enough for a batch
                        # or until we run out.
                        while len(data) < self.batch_max_size:
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
                await loop.run_in_executor(None, self.run, data[0] if not self.is_batched else data)
            except Exception as e:
                print("Exception in run_in_executor for %s:" % type(self), e)
                for result_future in result_futures:
                    if not result_future.cancelled():
                        result_future.set_exception(e)
                continue

            #print(type(self), "time: %.2fs" % (time() - step_start_time), "data count:", len(data))

            for result_future, datum in zip(result_futures, data):
                if not result_future.cancelled():
                    result_future.set_result(datum)

        print(colored("Task is finished", attrs=['bold']))


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

@asyncio.coroutine                                       
def exit():                                              
    loop = asyncio.get_event_loop()                      
    print("Stop")                                        
    loop.stop()                                          


def ask_exit():                          
    for task in asyncio.all_tasks():
        task.cancel()                    
    asyncio.ensure_future(exit()) 



from abc import ABCMeta, abstractmethod
from time import time, sleep
import typing
from multiprocessing import cpu_count
from enum import IntEnum
from termcolor import colored
from types import SimpleNamespace
import traceback
import threading, queue
from .config import PipelineMode, PipelineConfig

class PipelineStepIndex(IntEnum):
    Input = 0
    CalculateFov = 1
    PlaneDetector = 2
    EdgeDetector = 3
    Segmentation = 4
    ReverseRenderer = 5
    DeterminePrimaryAngles = 6
    ExtractSurfaces = 7
    FindLines = 8
    Geometry = 9
    GenerateScene = 10
    SolveSurfaces = 11
    VanishingPoints = 12
    Barriers = 13
    FindTrim = 14
    FindLegs = 15
    EstimatePose = 16
    Refine = 17
    Superpixels = 18
    CombinePlaneMasks = 19
    Output = 20

class PipelineStep(metaclass=ABCMeta):

    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.config = pipeline.config

        self.queue = queue.Queue()
        self.threads = []
        self.running = False

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

    def schedule(self, input_dict):
        self.queue.put(input_dict)

    def schedule_and_wait(self, input_dict):
        self.schedule(input_dict)
        #print("Waiting on task completion", self.queue.empty())
        self.queue.join() #await completion
        #print("completed")

    def start(self):
        if not self.running:
            self.running = True

            for _ in range(1 if self.is_batched else cpu_count()):
                thread = threading.Thread(target=self.thread_loop)
                self.threads.append(thread)
                thread.start()

    def stop(self):
        #print("Stopping %s" % self.description)

        self.running = False

        for thread in self.threads:
            thread.join()

        self.threads = []

        #print("%s is stopped" % self.description)

    def thread_loop(self):
        #print("thread %s started" % self.description)

        while self.running:
            try:
                if not self.queue.empty():
                    datum = self.queue.get_nowait()                    
                    self.run(datum)
                    self.queue.task_done()
            except:
                traceback.print_exc()
                pass

            sleep(0.1)

        #print("thread '%s' stopped" % self.description)
    

from abc import ABCMeta, abstractmethod, abstractproperty
from time import time
import asyncio
from multiprocessing import cpu_count


class PipelineStep(metaclass=ABCMeta):
    @abstractmethod
    def run(self, data: dict):
        pass

    @abstractproperty
    def required_keys(self) -> list:
        return {}

    @abstractproperty
    def output_keys(self) -> list:
        return {}

    @property
    def config(self) -> dict:
        return self._pipeline.config

    @property
    def is_batched(self) -> bool:
        return False


class Pipeline:
    def __init__(self, config={}):
        self._steps = []
        self._input_queue = asyncio.Queue()
        self._queues = [self._input_queue]
        self._queues_started = False
        self._config = config

    @property
    def config(self):
        return self._config

    @property
    def num_waiting_items(self):
        return sum([q.qsize() for q in self._queues])

    def add(self, step: PipelineStep):
        if self._queues_started:
            raise Exception("Can not add more pipeline steps after queues were started.")

        self._steps.append(step)
        step._pipeline = self
        return self

    def validate(self, keys: list):
        for step in self._steps:
            assert all(
                key in keys for key in step.required_keys), "One or more required keys missing for pipeline step %s" % step
            keys += step.output_keys

    def _start_queues(self):
        self._queues_started = True
        prev_queue = self._input_queue
        for i, step in enumerate(self._steps):
            step_queue = None
            if i + 1 < len(self._steps):
                step_queue = asyncio.Queue()
                self._queues.append(step_queue)

            for _ in range(1 if step.is_batched else cpu_count()):
                asyncio.ensure_future(Pipeline.run_step_in_background(step, prev_queue, step_queue))
                
            prev_queue = step_queue
        
    async def run(self, data: dict) -> dict:
        if not self._queues_started:
            self._start_queues()

        total_start_time = time()

        result_future = asyncio.get_event_loop().create_future()

        await self._input_queue.put((data, result_future))
        data = await result_future
        print("Total pipeline time: %.2fs" % (time() - total_start_time))

        return data

    @staticmethod
    async def run_step_in_background(step: PipelineStep, src_queue: asyncio.Queue, dst_queue: asyncio.Queue):
        loop = asyncio.get_event_loop()

        while True:
            datum, result_future = await src_queue.get()
            src_queue.task_done()

            if result_future.cancelled():
                continue

            result_futures = [result_future]
            data = [datum]
            
            if step.is_batched:
                debounce_start_time = time()
                wait_time = 0

                # Add items from the queue while more are available.
                # Do this by waiting a small amount of time for new data
                # up to a maximum time.
                while wait_time < 1.0 and len(data) < 4:
                    await asyncio.sleep(0.2)
                    try:
                        # Dequeue until we have enough for a batch
                        # or until we run out.
                        while len(data) < 4:
                            datum, result_future = src_queue.get_nowait()
                            if not result_future.cancelled():
                                result_futures.append(result_future)
                                data.append(datum)
                    except asyncio.QueueEmpty:
                        pass
                    wait_time = time() - debounce_start_time

            # Run the data
            step_start_time = time()
            try:
                await loop.run_in_executor(None, step.run, data[0] if not step.is_batched else data)
            except Exception as e:
                print("Exception in run_in_executor for %s:" % type(step), e)
                for result_future in result_futures:
                    if not result_future.cancelled():
                        result_future.set_exception(e)
                continue
                
            print(type(step), "time: %.2fs" % (time() - step_start_time), "data count:", len(data))

            if dst_queue is None:
                for result_future, datum in zip(result_futures, data):
                    if not result_future.cancelled():
                        result_future.set_result(datum)
            else:
                await asyncio.wait([dst_queue.put((datum, result_future)) for result_future, datum in zip(result_futures, data) if not result_future.cancelled()])

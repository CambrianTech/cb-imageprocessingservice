import multiprocessing as mp
import collections
import time
import psutil

Msg = collections.namedtuple('Msg', ['event', 'args'])

class BaseProcess(mp.Process):

    def __init__(self, input_queue, output_queue):
        super().__init__(target=self.run, args=())
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.running = True

    def dispatch(self, msg):
        event, args = msg

        handler = getattr(self, event, None)
        if not handler:
            raise NotImplementedError("Process has no handler for [%s]" % event)

        return handler(*args)

    def stop(self):
        print("Stop")
        self.running = False
        self.join()

    def run(self):

        while self.running:

            if self.input_queue.empty():
                time.sleep(0.01)
                continue

            msg = self.input_queue.get_nowait()
            result = self.dispatch(msg)
            self.output_queue.put(result)


class Multiprocessor():
    def __init__(self, subprocess_cls, num_workers=psutil.cpu_count()):
        self.subprocess_cls = subprocess_cls       
        self.input_queue = mp.Queue()
        self.output_queue = mp.Queue()
        self.num_tasks = 0

        self.subprocesses = []

        for i in range(num_workers):
            self.subprocesses.append(self.subprocess_cls(self.input_queue, self.output_queue))

    def start(self):
        for p in self.subprocesses:
            p.start()

    def schedule(self, event, *args):
        msg = Msg(event, args)
        self.input_queue.put(msg)
        self.num_tasks += 1

    def await_completion(self):
        completed_tasks_counter = 0
        results = []
        while completed_tasks_counter < self.num_tasks:
            result = self.output_queue.get()
            print("Got result", result)
            results.append(result)
            completed_tasks_counter += 1

        print("Finished")

        for p in self.subprocesses:
            p.join()

        return results
            
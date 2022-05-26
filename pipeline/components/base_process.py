import multiprocessing as mp
import collections
import time
import psutil
import queue
import traceback
import sys

ctx = mp.get_context('fork')

Msg = collections.namedtuple('Msg', ['event', 'args'])

SENTINEL = None
ERROR_THROWN = "error-thrown"

class BaseProcess(ctx.Process):

    def __init__(self, input_queue, output_queue):
        super().__init__(target=self.loop, args=(), daemon=True)
        self.input_queue = input_queue
        self.output_queue = output_queue

    def noop(self):
        pass

    def dispatch(self, msg):
        event, args = msg

        handler = getattr(self, event, None)
        if not handler:
            raise NotImplementedError("Process has no handler for [%s]" % event)

        return handler(*args)

    def loop(self):

        while True:

            try: 
                msg = self.input_queue.get_nowait()
                if msg == SENTINEL:
                    break

                result = self.dispatch(msg)
                self.output_queue.put(result)

            except queue.Empty:
                pass
            except:
                traceback.print_exc()
                self.output_queue.put(ERROR_THROWN)

            time.sleep(0.1)


class Multiprocessor():
    def __init__(self, subprocess_cls, num_workers=max(2, ctx.cpu_count()-1)):
        self.subprocess_cls = subprocess_cls
        self.input_queue = ctx.Queue()
        self.output_queue = ctx.Queue()
        
        self.num_workers = num_workers
        self.subprocesses = []

        self.num_tasks = 0

        for i in range(self.num_workers):
            self.subprocesses.append(self.subprocess_cls(self.input_queue, self.output_queue))

    def start(self):
        for p in self.subprocesses:
            p.start()

        #finish launching, this is necessary for "spawn" contexts only, which is default on windows/osx:
        for p in self.subprocesses:
            self.schedule("noop")

        self.await_completion()

    def schedule(self, event, *args):
        msg = Msg(event, args)
        self.input_queue.put_nowait(msg)
        self.num_tasks += 1

    def stop(self):
        for i in range(self.num_workers):
            self.input_queue.put_nowait(SENTINEL)

        for p in self.subprocesses:
            p.join()

    def kill(self):
        for i in range(self.num_workers):
            self.input_queue.put_nowait(SENTINEL)

    def schedule_and_wait(self, event, *args):
        self.schedule(event, *args)
        results = self.await_completion()

        return results[0]

    def await_completion(self):
        completed_tasks_counter = 0
        results = []
        while completed_tasks_counter < self.num_tasks:
            result = self.output_queue.get()

            if result == ERROR_THROWN:
                sys.exit('Multiprocessor process exited with error')

            results.append(result)
            completed_tasks_counter += 1

        self.num_tasks = 0
        
        return results
            
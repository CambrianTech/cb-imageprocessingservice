import requests
import numpy as np
import pickle
import time
import cv2

import os
import subprocess
import select
import threading
from termcolor import colored

from pipeline.core import PipelineStep, PipelineStepIndex
from pipeline.misc.utils import camera_fov_res_to_intrinsics, DaemonStoppableThread
from pipeline.misc.modelutils import feed_image_batched, load_model, get_session_config

class PipelineReverseRenderer(PipelineStep):

    def __init__(self, pipeline):
        super().__init__(pipeline)

        self.service_name = "%s:%d" % (self.config.cpu_networks_script, self.config.cpu_networks_port)
        self.child_process = None
        self.health_thread = None
        self.failed_attempts = 0
        self.debounce = 0.5
        self.models_loaded = False
        self.start_networks()
        #self.start_remote_process()

    @property
    def index(self) -> PipelineStepIndex:
        return PipelineStepIndex.ReverseRenderer

    @property
    def required_keys(self) -> list:
        return ["image"]

    @property
    def output_keys(self) -> list:
        return ["lighting", "normals"]

    @property
    def is_batched(self) -> bool:
        return True

    def healthchecker(self, endpoint="healthcheck"):  
        url = "%s/%s" % (self.config.cpu_networks_path, endpoint)
        try:
            # print("Posting data to %s" % url)
            resp = requests.post(url)
            resp.raise_for_status()

            if resp.ok:
                return
        except Exception as e:
            print(url, e)
            pass
        
        print("%s indicated dead process, restarting" % endpoint)
        self.stop_remote_process()

    def start_networks(self):
        print("Loading models from", self.config.model_path)
        self.model_lighting = load_model(os.path.join(self.config.model_path, "lighting"), session_config=self.config.session_config)
        self.model_normals = load_model(os.path.join(self.config.model_path, "normals"), session_config=self.config.session_config)

        self.models_loaded = True #todo:signal

    def start_remote_process(self, timeout=15, success_message = "$SUCCESS"):
        start = time.time()
        
        print(colored("Connecting to %s" % self.service_name, 'green'))

        self.child_process = subprocess.Popen(["python3", self.config.cpu_networks_script, self.config.model_path, str(self.config.cpu_networks_port), success_message], \
            stderr=subprocess.PIPE, universal_newlines=True)
        
        stream = self.child_process.stderr

        self.failed_attempts = 0
        success = False

        y = select.poll()
        y.register(stream, select.POLLIN)

        while time.time() - start < timeout:
            if y.poll(1):
                line = stream.readline()
                if len(line):
                    line = line.strip()
                    if line == success_message:
                        print(colored("Connection to %s successful. Received message: %s" % (self.service_name, line), 'green'))
                        success = True
                        break
                    elif self.config.cpu_networks_show_stderr:
                        print(self.service_name, line)
            else:
                time.sleep(self.debounce)

        if success: 
            time.sleep(self.debounce)
            print(colored("Starting healthcheck monitor"))
            self.health_thread = DaemonStoppableThread(sleep_time=self.config.cpu_networks_healthchecker_interval, target=self.healthchecker, name='health_thread')
            self.health_thread.start()
        else:
            print(colored("Connection to %s unsuccessful." % (self.service_name), 'red'))
            self.child_process = None


    def stop_remote_process(self):
        if self.child_process is None: return

        print("Killing subprocess")
        try:
            self.child_process.kill()
        except:
            print(colored("Warning: Python subprocess runcpunetworks.py already exited. It probably crashed!", 'yellow', attrs=['bold']))

        self.child_process = None

        if self.health_thread is not None and self.health_thread.is_alive():
            self.health_thread.stop()

    def remote_networks(self, images, endpoint="process"):
        url = "%s/%s" % (self.config.cpu_networks_path, endpoint)

        try:
            if self.child_process is None:
                self.start_remote_process() #todo: otherwise wait on thread signal 

            print("Running reverse renderer at", url)            

            resp = requests.post(url, data=pickle.dumps(images))
            resp.raise_for_status()

            if resp.ok:
                return pickle.loads(resp.content)

        except Exception as e:
            print(url, e)
            pass

        self.failed_attempts += 1

        if self.failed_attempts > self.config.cpu_networks_allotted_failures:
            print("Reverse renderer failed to connect to remote process for %d times beyond threshold %d" % (self.failed_attempts, self.config.cpu_networks_allotted_failures))
            self.stop_remote_process()

    def local_networks(self, images):

        print('Waiting on models to load')
        while not self.models_loaded:
            time.sleep(self.debounce)

        print('Models loaded')

        response_dict = {}

        print('Running lighting network')
        response_dict["lighting"] = feed_image_batched(self.model_lighting, images)

        print('Running normals network')
        response_dict["normals"] = feed_image_batched(self.model_normals, images)

        return response_dict

    def run(self, data: dict) -> None:

        images = [datum["image"] for datum in data]
        if len(images) == 0: 
            print("No images to reverse render")
            return   

        # response_dict = self.remote_networks(images)
        response_dict = self.local_networks(images)

        # if response_dict is None:
        #     return

        for datum, lighting, normals in zip(data, response_dict["lighting"], response_dict["normals"]):
            datum["lighting"] = lighting
            datum["normals"] = normals


    def stop(self):
        self.stop_remote_process()
            
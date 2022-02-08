import requests
import numpy as np
import pickle
import time
import cv2

import os
import subprocess
import select

from pipeline.core import PipelineStep, PipelineStepIndex
from pipeline.misc.utils import camera_fov_res_to_intrinsics
from termcolor import colored

class PipelineReverseRenderer(PipelineStep):

    def __init__(self, pipeline):
        super().__init__(pipeline)
        self.child_process = None

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

    def _remote_networks(self, data, timeout=15, debounce=0.5, success_message = "$SUCCESS"):
        if self.child_process is None:
            start = time.time()

            service_name = "%s:%d" % (self.config.cpu_networks_script, self.config.cpu_networks_port)
            print(colored("Connecting to %s" % service_name, 'green'))

            self.child_process = subprocess.Popen(["python3", self.config.cpu_networks_script, self.config.model_path, str(self.config.cpu_networks_port), success_message], \
                stderr=subprocess.PIPE, universal_newlines=True)
            
            stream = self.child_process.stderr

            y = select.poll()
            y.register(stream, select.POLLIN)

            while time.time() - start < timeout:
                if y.poll(1):
                    line = stream.readline()
                    if len(line):
                        line = line.strip()
                        if line == success_message:
                            print(colored("Connection to %s successful. Received message: %s" % (service_name, line), 'green'))
                            break
                else:
                    time.sleep(debounce)

        try:
            print("Posting data to %s" % self.config.cpu_networks_path)
            images = [datum["image"] for datum in data]
            resp = requests.post(self.config.cpu_networks_path, data=pickle.dumps(images))
            resp.raise_for_status()

            if resp.ok:
                return pickle.loads(resp.content)

        except requests.exceptions.HTTPError as e:
            error_message = e.response.text
            print(colored(error_message, 'red', attrs=['bold']))
        

    def run(self, data: dict) -> None:
        response_dict = self._remote_networks(data)

        if response_dict is None:
            return

        for datum, lighting, normals in zip(data, response_dict["lighting"], response_dict["normals"]):
            datum["lighting"] = lighting
            datum["normals"] = normals


    def stop(self):
        if self.child_process is None: return

        print("Killing subprocess")
        try:
            self.child_process.kill()
        except:
            print(colored("Warning: Python subprocess runcpunetworks.py already exited. It probably crashed!", 'yellow', attrs=['bold']))
            
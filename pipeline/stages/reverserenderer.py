import requests
import numpy as np
import pickle
from time import time
import cv2
import subprocess

from pipeline.core import PipelineStep, PipelineStepIndex
from pipeline.misc.utils import camera_fov_res_to_intrinsics
from termcolor import colored

class PipelineReverseRenderer(PipelineStep):

    def __init__(self, pipeline):
        super().__init__(pipeline)
        self.child_process = _start_process(self.config)

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

    def _remote_networks(self, data):
        print("Posting data to %s" % config.cpu_networks_path)
        images = [datum["image"] for datum in data]
        response_bytes = requests.post(config.cpu_networks_path, data=pickle.dumps(images)).content
        return pickle.loads(response_bytes)

    def _start_process(self):
        #start RemoteNetworks if needed downstream
        print("Listening on port ", config.cpu_networks_port, "runcpunetworks.py")
        return subprocess.Popen(["python3", "runcpunetworks.py", config.model_path, str(config.cpu_networks_port)])

    def run(self, data: dict) -> None:
        t = time()
        response_dict = _remote_networks(self.config, data)
        print("Remote networks took %.2f seconds" % (time() - t))

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
            
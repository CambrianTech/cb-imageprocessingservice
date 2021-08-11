import os
import numpy as np
from pathlib import Path
import pickle

import click
import time

from pipeline.core import schedule_and_wait, merge_future_dicts, num_waiting_items
from pipeline.fov import PipelineCalculateFov
from pipeline.getdata import PipelineGetData
from pipeline.primaryangle import PipelineDeterminePrimaryAngles
from pipeline.runmodels import PipelineRunModels
from pipeline.superpixels import PipelineSuperpixels
from pipeline.refineplanemasks import PipelineRefinePlaneMasks
from pipeline.combineplanemasks import PipelineCombinePlaneMasks
from pipeline.uploadresults import PipelineUploadResults
from pipeline.remote import PipelineRemotePlaneDetector, PipelineRemoteNetworks

def get_file_paths(input_dir, pattern="*.pickle"):
    files = []
    files.extend(Path(input_dir).glob('**/' + pattern))
            
    return files

def parse_data(input_dir, output_dir):
    files = get_file_paths(input_dir, "*.pickle")

    print("Importing %d files from \"%s\" into \"%s\"" % (len(files), input_dir, output_dir))

    for path in files:
        data_path = str(path)
        dir_path = os.path.dirname(data_path)

        print("Parsing", path)

        with open(path, 'rb') as handle:
            data_pickle = pickle.load(handle)
            run_harness(data_pickle, dir_path)

    index = 0

    return len(files)

def run_harness(data, directory):
    print("Processing %s" % directory, data.shape)


@click.command()
@click.argument("input_dir", default='test_images', type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.argument("output_dir", default='output', type=click.Path(exists=False, file_okay=False, dir_okay=True))
@click.argument("model_path", type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.argument("semantic_model_path", type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.argument("fov_model_path", type=click.Path(exists=True, file_okay=True, dir_okay=False))
def main(input_dir, output_dir, model_path, semantic_model_path, fov_model_path):

    if not os.path.exists(input_dir):
        raise Exception('The directory does not exist at path {}'.format(input_dir)) 

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    start = time.time()

    num_files = parse_data(input_dir, output_dir)
    elapsed = (time.time() - start)

    if num_files > 0:
        print("Processing %d data took %.2f seconds (%.2fs each)" % (num_files, elapsed, elapsed/num_files))

if __name__ == "__main__":
    main()


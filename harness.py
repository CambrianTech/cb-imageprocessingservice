import os
import numpy as np
from pathlib import Path
import pickle

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

import click
import time
import asyncio
import signal
from concurrent.futures import ThreadPoolExecutor
from termcolor import colored

from pipeline.core import ask_exit, PipelineMode, PipelineConfig, PipelineStepIndex
from pipeline.pipeline import Pipeline
from pipeline.data.logging import LogLevel

def get_file_paths(input_dir, pattern=None):
    files = []
    if pattern:
        files.extend(Path(input_dir).glob('**/' + pattern))
    else: 
        extensions = ('.png', '.jpg', '.jpeg')
        for ext in extensions:
            files.extend(Path(input_dir).glob('**/*' + ext))
    return files

async def process_files(pipeline, files):

    index = 1
    
    for path in files:
        url = Path(path)
        unique_id = url.parents[0].name if len(url.parents) > 0 else url.name
        data = {"path": path, "unique_id": unique_id if path.suffix == ".pickle" else url.stem}

        print("\nProcessing file %d of %d\n" % (index, len(files)))
        await pipeline.process(data)
        index += 1


#For instance, to restore from step 6 (before refinement):
#python -W ignore harness.py data --restore=6

default_config = PipelineConfig()

@click.command()
@click.argument("input_dir", default='test_images', type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.argument("output_dir", default='output', type=click.Path(exists=False, file_okay=False, dir_okay=True))
@click.argument("model_path", default=default_config.model_path, type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.argument("semantic_model_path", default=default_config.semantic_model_path, type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.argument("fov_model_path", default=default_config.fov_model_path, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.argument("hed_model_path", default=default_config.hed_model_path, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.argument("planes_url", default=default_config.planes_url, type=click.STRING)
@click.option('--api', type=int, default=default_config.api_level, help='api level: 1-4')
@click.option('--restore', type=int, default=default_config.restore_step, help='Pipeline step to restore from. Data pickle files expected inside input_dir')
@click.option('--export', type=int, default=default_config.export_step, help='Pipeline step to export')
@click.option('--stop', type=int, default=default_config.stop_step, help='Stop after step')
@click.option("--log_dir", type=click.Path(exists=False, file_okay=False, dir_okay=True), default='logging')
@click.option('--log_level', type=int, default=LogLevel.Default, help='corresponds to LogLevel inside pipeline/logging, a binary mask: models | segmentation | images, default All')
@click.option('--log_step', type=int, default=default_config.logging_step, help='Log only a single step in the pipeline')
def main(input_dir, output_dir, model_path, semantic_model_path, fov_model_path, hed_model_path, planes_url, \
         api, restore, export, stop, log_dir, log_level, log_step):

    if not os.path.exists(input_dir):
        raise Exception('The directory does not exist at path {}'.format(input_dir)) 

    
    file_pattern = "*.pickle" if restore is not None else None

    files = get_file_paths(input_dir, file_pattern)

    def print_title(text):
        print(colored("\n%s\n" % text, 'blue', 'on_white', attrs=['bold']))
        print("\n")

    if file_pattern is None:
        print_title("Processing %d images from \"%s\"" % (len(files), input_dir))
    else:
        print_title("Processing %d files from \"%s/**/%s\"" % (len(files), input_dir, file_pattern))

    if len(files) == 0:
        raise Exception('No files found at path {}'.format(input_dir)) 
    
    config = PipelineConfig()
    config.mode = PipelineMode.Restore if restore is not None else PipelineMode.Process
    config.src_path=input_dir
    config.dest_path=output_dir

    config.model_path = model_path
    config.semantic_model_path = semantic_model_path
    config.fov_model_path = fov_model_path
    config.hed_model_path = hed_model_path
    config.planes_url = planes_url

    config.api_level = api
    config.restore_step = None if restore is None else PipelineStepIndex(restore)
    config.export_step = None if export is None else PipelineStepIndex(export)
    config.stop_step = None if stop is None else PipelineStepIndex(stop)
    config.logging_dir = log_dir
    config.logging_level = log_level
    config.logging_step = None if log_step is None else PipelineStepIndex(log_step)

    pipeline = Pipeline(config)


    loop = asyncio.get_event_loop()
    loop.set_default_executor(ThreadPoolExecutor())

    for sig in (signal.SIGINT, signal.SIGTERM):          
        loop.add_signal_handler(sig, ask_exit)  

    start_time = time.time()

    pipeline.start()
    loop.run_until_complete(process_files(pipeline, files))
    pipeline.stop()

    elapsed = (time.time() - start_time)
    avg = elapsed / len(files)
    
    print_title("Total processing time: %.2fs, average: %.2fs" % (elapsed, avg))

if __name__ == "__main__":
    main()


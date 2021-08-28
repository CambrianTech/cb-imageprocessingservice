import os
import numpy as np
from pathlib import Path
import pickle

import click
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor

from pipeline.pipeline import Pipeline, PipelineMode, PipelineStepIndex
from pipeline.logging import LogLevel

def get_file_paths(input_dir, pattern=None):
    files = []
    if pattern:
        files.extend(Path(input_dir).glob('**/' + pattern))
    else: 
        extensions = ('.png', '.jpg', '.jpeg')
        for ext in extensions:
            files.extend(Path(input_dir).glob('**/*' + ext))
    return files

async def process_files(pipeline, input_dir, pattern):
    files = get_file_paths(input_dir, pattern)

    print("Importing %d files from \"%s\"" % (len(files), input_dir))

    for path in files:
        url = Path(path)
        unique_id = url.parents[0].name
        data = {"path": path, "unique_id": unique_id if path.suffix == ".pickle" else url.stem}

        await pipeline.process(data)

    index = 0

    return len(files)


#For instance, to restore from step 7 (after refinement):
#python -W ignore harness.py data --restore=7

@click.command()
@click.argument("input_dir", default='test_images', type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.argument("output_dir", default='output', type=click.Path(exists=False, file_okay=False, dir_okay=True))
@click.argument("model_path", default='tensorflow_models', type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.argument("semantic_model_path", default='gluon_models', type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.argument("fov_model_path", default='sklearn_models/fov_classifier_lc128.joblib', type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.argument("planes_url", default='http://localhost:8081/', type=click.STRING)
@click.option('--api_level', type=int, default=4, help='api level: 1-4')
@click.option('--restore', type=int, help='Pipeline step to restore from. Data pickle files expected inside input_dir')
@click.option('--export', type=int, help='Pipeline step to export')
@click.option("--logging_dir", type=click.Path(exists=False, file_okay=False, dir_okay=True), default='logging')
@click.option('--log_level', type=int, default=LogLevel.Images|LogLevel.Segmentation, help='corresponds to LogLevel inside pipeline/logging, a binary mask: models | segmentation | images, default All')
@click.option('--log_step', type=int, default=None, help='Log only a single step in the pipeline')
def main(input_dir, output_dir, model_path, semantic_model_path, fov_model_path, planes_url, 
         api_level, restore, export, logging_dir, log_level, log_step):

    if not os.path.exists(input_dir):
        raise Exception('The directory does not exist at path {}'.format(input_dir)) 

    
    mode = PipelineMode.Restore if restore is not None else PipelineMode.Process

    file_pattern = "*.pickle" if restore is not None else None
    restore_step = PipelineStepIndex(restore) if restore is not None else None
    export_step = PipelineStepIndex(export) if export is not None else None
    logging_step = PipelineStepIndex(log_step) if log_step is not None else None

    loop = asyncio.get_event_loop()
    loop.set_default_executor(ThreadPoolExecutor())

    pipeline = Pipeline(mode, api_level, src_path=input_dir, dest_path=output_dir, restore_step=restore_step, export_step=export_step, logging_dir=logging_dir, logging_level=log_level, logging_step=logging_step, \
                        model_path=model_path, semantic_model_path=semantic_model_path, fov_model_path=fov_model_path, planes_url=planes_url)
    pipeline.start()

    loop.run_until_complete(process_files(pipeline, input_dir, pattern=file_pattern))
    


if __name__ == "__main__":
    main()


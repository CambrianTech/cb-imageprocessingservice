import os
import numpy as np
from pathlib import Path
import pickle

import click
import time
import asyncio

from pipeline.buildpipeline import Pipeline

def get_file_paths(input_dir, pattern=None):
    files = []
    if pattern:
        files.extend(Path(input_dir).glob('**/' + pattern))
    else: 
        extensions = ('.png', '.jpg', '.jpeg')
        for ext in extensions:
            files.extend(Path(input_dir).glob('**/*' + ext))
    return files

async def process_files(pipeline, input_dir, output_dir):
    files = get_file_paths(input_dir)

    print("Importing %d files from \"%s\" into \"%s\"" % (len(files), input_dir, output_dir))

    for path in files:
        data_path = str(path)
        dir_path = os.path.dirname(data_path)

        data = {"image_path": path}

        print("Processing", path)

        await pipeline.process(data)

    index = 0

    return len(files)


@click.command()
@click.argument("input_dir", default='test_images', type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.argument("output_dir", default='output', type=click.Path(exists=False, file_okay=False, dir_okay=True))
@click.argument("model_path", default='tensorflow_models', type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.argument("semantic_model_path", default='gluon_models', type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.argument("fov_model_path", default='sklearn_models/fov_classifier_lc128.joblib', type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.argument("planes_url", default='http://planes:8081/', type=click.STRING)

def main(input_dir, output_dir, model_path, semantic_model_path, fov_model_path, planes_url):

    if not os.path.exists(input_dir):
        raise Exception('The directory does not exist at path {}'.format(input_dir)) 

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    pipeline = Pipeline(model_path, semantic_model_path, fov_model_path, planes_url)
    pipeline.start()

    loop = asyncio.get_event_loop()
    loop.run_until_complete(process_files(pipeline, input_dir, output_dir))
    loop.close()


if __name__ == "__main__":
    main()


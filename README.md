# Cambrian Image Processing Service

This service performs image processing operations such as semantic segmentation and exposes them as Http REST APIs. It can be built into a docker image by running `docker build .`. It will execute `start.sh` on startup. Furthermore unit tests can be run with eg. `python -m unittest discover tests`.

# REST API
## `GET` `/segment/{S3 Key}/`
- `{S3 Key}` is the key for the source image
- Returns `{ "lighting_url": "https://url/to/lighting.png, "semantic_url": "https://url/to/semantic.png", "fov": <fov in degrees> }`

# Architecture
The system is based on a pipeline where multiple pipeline steps are performed sequentially. The steps are passed a dictionary which they can read from and write to. Initially the only key is `unique_id`. In the end the `lighting_url` and `semantic_url` entries are used for uploading the images to S3.

# Installation

## Models and weights
The image processing server uses tensorflow models for CNNs as well as a scikit-learn model for estimating field of view. The models can be copied from `cb-deepweb`.
The directory with the tensorflow models should contain the following subdirectories: `elevation`, `lighting`, `normals`, `semantic`, `unlit`.

Example:
```
cb-imageprocessingservice
  tensorflow_models
    elevation
    lighting (renamed from shadows)
    normals
    semantic
    unlit
  sklearn_models
    fov_classifier_lc128.joblib 
```

## Prerequisites OSX
pygobject3: `brew install pygobject3`

## Installing prerequisites into a Virtualenv
Virtualenv: `pip3 install virtualenv`

Use a virtual environment and utilize the requirements.txt or if OSX darwin-requirements.txt in this package as well as the cb-core repository:
```
virtualenv ~/venv/shaw
source ~/venv/shaw/bin/activate
pip3 install -r REQ-TXT-FILE
pip3 install -e PATH_TO_CB_CORE_REPO
```

## Running the server
The `serve.py` script can be run to start the http image processing server:
`python3 serve.py <tf_models_path> <fov_model_path> <s3_images_bucket> <s3_results_bucket>`

Optionally `--images-local-dir <path>` and `--results-local-dir <path>` can be passed to read from and write to these paths instead of S3 for local testing.

Example: `python3 serve.py ./tensorflow_models/ ./sklearn_models/fov_classifier_lc128.joblib cb-user-image-uploads cb-imageprocessingservice-results --image-local-dir ./images/ --results-local-dir ./results/`

## Developing locally
install docker and cuda docker
- run buildlocal.sh to build the plane and backend image
- run startlocal.sh to start the two containers and connect them together etc.

you can put images into the images folder on the backend server (use docker cp to copy files from or to the container), and then you can run localhost:8081/segment/<name of image> to segment those, and results will be in the results folder on the container (use docker cp to copy those out again)

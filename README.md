# Cambrian Image Processing Service

This service performs image processing operations such as semantic segmentation and exposes them as Http REST APIs. It can be built into a docker image by running `docker build .`. It will execute `start.sh` on startup. Furthermore unit tests can be run with eg. `python -m unittest discover tests`.

# Rest API
## GET segment/{S3 Key}/
- `{S3 Key}` is the key for the source image
- Returns `{ "lighting_url": "https://url/to/lighting.png, "semantic_url": "https://url/to/semantic.png" }`

# Architecture
The system is based on a pipeline where multiple pipeline steps are performed sequentially. The steps are passed a dictionary which they can read from and write to. Initially the only key is `image_s3_key`. In the end the `lighting_url` and `semantic_url` entries are used for uploading the images to S3.
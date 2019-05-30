#!/bin/sh

echo "Downloading S3 data"
aws s3 cp s3://cb-imageprocessingservice . --recursive

echo "Starting python serve script"
python3 serve.py ./tensorflow_models/ ./sklearn_models/fov_classifier_lc128.joblib cb-user-image-uploads cb-imageprocessingservice-results "$@"
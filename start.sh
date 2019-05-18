#!/bin/sh

echo "Downloading S3 data"
aws s3 cp s3://cb-imageprocessingservice . --recursive

echo "Starting python serve script"
python3 serve.py ./tensorflow_models/ ./sklearn_models/ cb-user-image-uploads cb-imageprocessingservice-results
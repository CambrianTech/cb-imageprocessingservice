#!/bin/sh

#FILES_BUCKET=cb-imageprocessingservice-models
#echo "Downloading S3 data from ${FILES_BUCKET}"
#aws s3 cp s3://$FILES_BUCKET . --recursive

echo "Starting python serve script"
python3 serve.py ./tensorflow_models/ ./gluon_models/ ./sklearn_models/fov_classifier_lc128.joblib $USER_UPLOADS_BUCKET $RESULTS_BUCKET $PLANES_ADDRESS
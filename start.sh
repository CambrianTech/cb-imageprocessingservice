#!/bin/sh
echo USER_UPLOADS_BUCKET: $USER_UPLOADS_BUCKET
echo RESULTS_BUCKET: $RESULTS_BUCKET
echo PLANES_ADDRESS: $PLANES_ADDRESS
echo SQS_QUEUE_NAME: $SQS_QUEUE_NAME

echo "Downloading S3 data"
aws s3 cp s3://$FILES_BUCKET . --recursive

export MXNET_CUDNN_AUTOTUNE_DEFAULT=0

echo "Starting python serve script"
python3 serve.py ./tensorflow_models/ ./gluon_models/ ./sklearn_models/fov_classifier_lc128.joblib ./hed_model/HED_pretrained_bsds.npz $USER_UPLOADS_BUCKET $RESULTS_BUCKET $PLANES_ADDRESS --sqs-queue-name $SQS_QUEUE_NAME --export=5
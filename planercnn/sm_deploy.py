from sagemaker.pytorch.model import PyTorchModel

role = "arn:aws:iam::312943975091:role/service-role/AmazonSageMaker-ExecutionRole-20200716T135671"

model = PyTorchModel(
    model_data="s3://sagemaker-us-east-1-312943975091/planercnn/model.tar.gz",
    source_dir="s3://sagemaker-us-east-1-312943975091/planercnn/sourcedir.tar.gz",
    role=role,
    entry_point="sm_main.py",
    framework_version="0.4.0",
    image_uri="312943975091.dkr.ecr.us-east-1.amazonaws.com/cb-planercnn-sagemaker:latest"
)

model.deploy(
    initial_instance_count=1,
    instance_type="ml.g4dn.xlarge"
)

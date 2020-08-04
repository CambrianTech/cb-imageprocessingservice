from sagemaker.pytorch.model import PyTorchModel


model = PyTorchModel(
    model_data="s3://",
    source_dir="",
    role="",
    entry_point="sm_main.py",
    framework_version="0.4.1"
)

model.deploy(
    instance_type="ml.g4dn.xlarge",
    initial_instance_count=1
)

import os
import sagemaker
from sagemaker.tensorflow import TensorFlow

role = os.getenv("SAGEMAKER_EXECUTION_ROLE")
if not role:
    raise ValueError("SAGEMAKER_EXECUTION_ROLE is not set")

sagemaker_session = sagemaker.Session()

input_dir = "input/data/train"
s3_input = sagemaker_session.upload_data(path=input_dir, key_prefix="dressmeai/train")

estimator = TensorFlow(
    entry_point='train_runner.py',
    source_dir='src',
    role=role,
    instance_count=1,
    instance_type='ml.m5.large',
    framework_version='2.18.0',
    py_version='py310',
    script_mode=True,
    output_path=f"s3://{sagemaker_session.default_bucket()}/dressmeai/output",
    base_job_name="dressmeai-train",
    hyperparameters={
        'epochs': 1000,
    },
)

estimator.fit({"training": s3_input})


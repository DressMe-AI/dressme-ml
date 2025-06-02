import sagemaker
from sagemaker.tensorflow import TensorFlow
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

role = os.getenv("SAGEMAKER_EXECUTION_ROLE")

# Set up SageMaker session and role
sagemaker_session = sagemaker.Session()

# Upload your training data
input_dir = "input/data/train"
s3_input = sagemaker_session.upload_data(path=input_dir, key_prefix="dressmeai/train")

# Define the estimator
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
        'epochs': 10,
    },
)

# Launch training
estimator.fit({"training": s3_input})

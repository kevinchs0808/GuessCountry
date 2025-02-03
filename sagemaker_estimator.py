from sagemaker import get_execution_role
from sagemaker.ray import RayEstimator

role = get_execution_role()

# Define Ray parameters
ray_params = {
    "num_workers": 2,  # Number of workers for the Ray cluster
    "resources_per_worker": {"CPU": 1},  # Each worker gets 1 CPU
    "hyperparameters": {
        "epochs": 100,
        "embedding_dim": 64,
        "lr": 0.01
    }
}

# Define the Ray Estimator
ray_estimator = RayEstimator(
    entry_point="train.py",  # Entry script for training
    role=role,
    instance_count=2,
    instance_type="ml.m5.large",  # Choose an appropriate instance type
    ray_params=ray_params,
    source_dir="s3://my-ray-gnn-bucket/project/",  # S3 path to the project directory
    framework_version="2.0.0",  # Define Ray version
    sagemaker_session=sagemaker.Session(),
    output_path="s3://my-ray-gnn-bucket/output",  # Path to save the trained model
    dependencies=["torch", "torch-geometric", "ray", "pandas", "matplotlib", "seaborn"]
)

# Start training job
ray_estimator.fit({"train": "s3://my-ray-gnn-bucket/data"})
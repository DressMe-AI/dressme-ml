import argparse
import logging
import os
from train import run_training_pipeline

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Run training pipeline.")
    parser.add_argument('--data_dir', type=str, required=True, help="Directory containing input data")
    parser.add_argument('--model_dir', type=str, required=True, help="Directory to store final model for SageMaker")
    parser.add_argument('--epochs', type=int, default=20, help="Number of training epochs")
    return parser.parse_args()

if __name__ == "__main__":
    try:
        args = parse_args()
        logger.info(f"Starting training: data_dir={args.data_dir}, output_dir={args.output_dir}, epochs={args.epochs}")
        run_training_pipeline(args.data_dir, args.model__dir, args.epochs)
    except Exception as e:
        logger.exception(f"Fatal error during training: {e}")
        raise

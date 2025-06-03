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
    parser.add_argument('--data_dir', type=str, default=os.environ.get("SM_CHANNEL_TRAINING", "/opt/ml/input/data/training"))
    parser.add_argument('--model_dir', type=str, default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    parser.add_argument('--epochs', type=int, default=20, help="Number of training epochs")
    return parser.parse_args()

if __name__ == "__main__":
    try:
        args = parse_args()
        model_dir_local = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
        logger.info(f"Begin Train: data_dir={args.data_dir}, model_dir={model_dir_local}, epochs={args.epochs}")
        run_training_pipeline(args.data_dir, model_dir_local, args.epochs)
    except Exception as e:
        logger.exception(f"Fatal error during training: {e}")
        raise

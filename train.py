import argparse
import os
import json
from utils import import_attributes, call_data, train_validate_model, train_final_model
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser()
    logger.info(f"Starting training: data_dir={args.data_dir}, output_dir={args.output_dir}, epochs={args.epochs}")

    # SageMaker provides these by default
    parser.add_argument('--output_dir', type=str, default=os.environ.get('SM_MODEL_DIR', '/opt/ml/model'))
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAINING', '/opt/ml/input/data/train'))
    parser.add_argument('--epochs', type=int, default=20)  # fallback

    return parser.parse_args()


def main():
    args = parse_args()
    assert os.path.isdir(args.data_dir), f"Provided data_dir does not exist: {args.data_dir}"
    assert isinstance(args.epochs, int) and args.epochs > 0, f"Epochs must be a positive integer, got: {args.epochs}"
    
    # Load data
    encoded_df = import_attributes(args.data_dir)
    assert not encoded_df.empty, "Encoded attributes DataFrame is empty."
    logger.info("Attributes loaded and encoded successfully.")

    X, y = call_data(encoded_df, args.data_dir)
    assert X.shape[0] > 1, "Need at least 2 training samples to split train/val."
    assert X.shape[0] == y.shape[0], "Mismatch between features and labels."
    logger.info(f"Training data prepared: X.shape={X.shape}, y.shape={y.shape}")

    best_epoch = train_validate_model(X, y, epochs=args.epochs, verbose=True)
    assert isinstance(best_epoch, int) and best_epoch > 0, f"Invalid best_epoch: {best_epoch}"
    logger.info(f"Best epoch selected from validation: {best_epoch}")

    os.makedirs(args.output_dir, exist_ok=True)

    train_final_model(
        X, y,
        best_epoch=best_epoch,
        tflite_path=os.path.join(args.output_dir, "model.tflite")
    )
    assert os.path.exists(tflite_model_path), "TFLite model was not saved."
    assert os.path.getsize(tflite_model_path) > 0, "TFLite model file is empty."
    logger.info("Final model trained and converted to TFLite.")

    # Move logs to output dir
    for log_file in ["training_log.csv", "final_training_log.csv", "best_model.weights.h5"]:
        if os.path.exists(log_file):
            os.rename(log_file, os.path.join(args.output_dir, log_file))
    logger.info("Training completed successfully. Logs and model saved to output_dir.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception(f"Fatal error during training: {e}")
        raise

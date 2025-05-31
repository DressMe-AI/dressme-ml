import argparse
import os
import json

from utils import import_attributes, call_data, train_validate_model, train_final_model

def parse_args():
    parser = argparse.ArgumentParser()

    # SageMaker provides these by default
    parser.add_argument('--output_dir', type=str, default=os.environ.get('SM_MODEL_DIR', '/opt/ml/model'))
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAINING', '/opt/ml/input/data/train'))
    parser.add_argument('--epochs', type=int, default=20)  # fallback

    return parser.parse_args()


def main():
    args = parse_args()

    # Load data
    encoded_df = import_attributes(args.data_dir)
    X, y = call_data(encoded_df, args.data_dir)

    best_epoch = train_validate_model(X, y, epochs=args.epochs, verbose=True)

    os.makedirs(args.output_dir, exist_ok=True)

    train_final_model(
        X, y,
        best_epoch=best_epoch,
        tflite_path=os.path.join(args.output_dir, "model.tflite")
    )

    # Move logs to output dir
    for log_file in ["training_log.csv", "final_training_log.csv", "best_model.weights.h5"]:
        if os.path.exists(log_file):
            os.rename(log_file, os.path.join(args.output_dir, log_file))


if __name__ == "__main__":
    main()

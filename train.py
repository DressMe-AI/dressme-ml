# train.py

import argparse
import os
from utils import import_attributes, call_data, train_validate_model, train_final_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/opt/ml/input/data/train")
    parser.add_argument("--output_dir", type=str, default="/opt/ml/model")
    args = parser.parse_args()

    encoded_df = import_attributes(args.data_dir)
    X, y = call_data(encoded_df, args.data_dir)

    best_epoch = train_validate_model(X, y, verbose=True)

    os.makedirs(args.output_dir, exist_ok=True)

    # Train and save model
    train_final_model(
        X, y,
        best_epoch=best_epoch,
        tflite_path=os.path.join(args.output_dir, "model.tflite")
    )

    # Move logs to model dir
    for log_file in ["training_log.csv", "final_training_log.csv", "best_model.weights.h5"]:
        if os.path.exists(log_file):
            os.rename(log_file, os.path.join(args.output_dir, log_file))


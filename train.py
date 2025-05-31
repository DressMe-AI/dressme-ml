# train.py

import argparse
import os
from utils import import_attributes, call_data, train_validate_model, train_final_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/opt/ml/input/data/train")
    parser.add_argument("--mode", choices=["validate", "final"], default="final")
    parser.add_argument("--output_dir", type=str, default="/opt/ml/model")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    encoded_df = import_attributes(args.data_dir)
    X, y = call_data(encoded_df, args.data_dir)

    if args.mode == "validate":
        best_epoch = train_validate_model(X, y, verbose=True)
    else:
        best_epoch = args.epochs or 100
        train_final_model(X, y, best_epoch, tflite_path=os.path.join(args.output_dir, "model.tflite"))

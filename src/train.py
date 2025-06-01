import os
from utils import import_attributes, call_data, train_validate_model, train_final_model
import logging

logger = logging.getLogger(__name__)

def run_training_pipeline(data_dir: str, model_dir: str, epochs: int):
    assert os.path.isdir(data_dir), f"Provided data_dir does not exist: {data_dir}"
    assert isinstance(epochs, int) and epochs > 0, f"Epochs must be a positive integer, got: {epochs}"

    encoded_df = import_attributes(data_dir)
    assert not encoded_df.empty, "Encoded attributes DataFrame is empty."
    logger.info("Attributes loaded and encoded successfully.")

    X, y = call_data(encoded_df, data_dir)
    assert X.shape[0] > 1, "Need at least 2 training samples to split train/val."
    assert X.shape[0] == y.shape[0], "Mismatch between features and labels."
    logger.info(f"Training data prepared: X.shape={X.shape}, y.shape={y.shape}")

    best_epoch = train_validate_model(X, y, epochs=epochs)
    assert isinstance(best_epoch, int) and best_epoch > 0, f"Invalid best_epoch: {best_epoch}"
    logger.info(f"Best epoch selected from validation: {best_epoch}")

    os.makedirs(model_dir, exist_ok=True)
    tflite_model_path = os.path.join(model_dir, "model.tflite")

    train_final_model(X, y, best_epoch=best_epoch, tflite_path=tflite_model_path)

    assert os.path.exists(tflite_model_path), "TFLite model was not saved."
    assert os.path.getsize(tflite_model_path) > 0, "TFLite model file is empty."
    logger.info("Final model trained and converted to TFLite.")

    for log_file in ["training_log.csv", "final_training_log.csv", "best_model.weights.h5"]:
        if os.path.exists(log_file):
            os.rename(log_file, os.path.join(model_dir, log_file))

    logger.info("Training completed successfully. Logs and model saved to model_dir.")

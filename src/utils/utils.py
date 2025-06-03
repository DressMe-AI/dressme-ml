import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
import logging

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)


def import_attributes(attributes_path: str) -> pd.DataFrame:
    """
    Load clothing attributes from a JSON file and encode categorical features numerically.

    Args:
        attributes_path (str): Path to the directory containing 'attributes_new.json'.

    Returns:
        pd.DataFrame: Encoded DataFrame with numerical representations of attributes.
    """
    
    # Load attributes
    with open(os.path.join(attributes_path, "attributes.json"), "r") as f:
        attributes = json.load(f)

    # Convert to DataFrame
    df = pd.DataFrame(attributes)

    # Define mappings for each categorical column
    mappings = {
      "type": {"top": 0, "bottom": 1},
      "color1": {"red": 0, "blue": 1, "white": 2, "black": 3, "brown": 4, "green": 5, "yellow": 6,
                 "gray": 7, "navy": 8, "pink": 9},
      "color2": {"red": 0, "blue": 1, "white": 2, "black": 3, "brown": 4, "green": 5, "yellow": 6, 
                 "gray": 7, "navy": 8, "pink": 9, "none": 10},
      "pattern": {"solid": 0, "striped": 1, "floral": 2, "plaid": 3, "polka dot": 4},
      "dress_code": {"formal": 0, "casual": 1},
      "material": {"cotton": 0, "denim": 1, "silk": 2, "wool": 3, "linen": 4, "polyester": 5, 
                   "unknown": 6},
      "seasonality": {"spring": 0, "summer": 1, "fall": 2, "winter": 3, "all": 4},
      "fit": {"loose": 0, "relaxed": 1, "fitted": 2, "tailored": 3, "slim": 4}
    }

    # Create a new DataFrame for numerical representation
    encoded_df = df.copy()
    for column, mapping in mappings.items():
        encoded_df[column] = df[column].map(mapping)
        # Handle unexpected values by raising a warning
        if encoded_df[column].isna().any():
            logger.warning(f"Unknown values in '{column}': {df[column][encoded_df[column].isna()].unique()}")

    return encoded_df


def call_data(encoded_df: pd.DataFrame, combinations_path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load clothing combinations and labels from file and convert them into feature arrays.

    Args:
        encoded_df (pd.DataFrame): DataFrame containing encoded clothing attributes.
        combinations_path (str): Path to the directory containing 'combination_scored.txt'.

    Returns:
        tuple:
            - X (np.ndarray): Feature matrix of shape (n_samples, n_features).
            - y (np.ndarray): Binary labels (0 or 1).
    """

    # Load combinations and scores
    with open(os.path.join(combinations_path, "combination_scored.txt"), "r") as f:
        combinations = [line.strip() for line in f]

    X = []
    y = []

    for combo in combinations:
        parts = combo.split(",")
        top_id = parts[0].split(":")[1]
        bottom_id = parts[1].split(":")[1]
        score = int(parts[2])

        # Get encoded attributes for top and bottom (Only picked related content.)
        top_attrs = encoded_df[encoded_df["id"] == top_id][
["color1", "pattern", "material", "fit"]
].values
        bottom_attrs = encoded_df[encoded_df["id"] == bottom_id][
["color1", "pattern", "material", "fit"]
].values
        top_attrs = encoded_df[encoded_df["id"] == top_id][["color1", "pattern", 
                                                            "material","fit"]].values
        bottom_attrs = encoded_df[encoded_df["id"] == bottom_id][["color1", "pattern", 
                                                                  "material", "fit"]].values
        
        if top_attrs.size == 4 and bottom_attrs.size == 4:
            combo_attrs = np.stack([top_attrs[0], bottom_attrs[0]], axis=-1)  # Shape: (6,2)
            X.append(combo_attrs)
            y.append(score)

    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)

    mask = y != 0  # Ignore y == 0 #Ones the user is neutral about.
    X = X[mask]
    y = y[mask]
    X = X.reshape(X.shape[0], -1)
    y = np.array([0 if score == -1 else 1 for score in y])

    logger.info(f"Total training pairs loaded: {X.shape[0]}")
    logger.info(f"Feature dimensions per sample: {X.shape[1]}")

    return X, y


def train_validate_model(X: np.ndarray, y: np.ndarray,
                         epochs: int = 100,
                         checkpoint_path: str = "best_model.weights.h5",
                         seed: int = 42) -> int:
    """
    Train a binary classifier using the provided dataset, validate it,
    and return the epoch that gives the highest validation accuracy.

    Args:
        X (np.ndarray): Feature matrix of shape (n_samples, n_features).
        y (np.ndarray): Binary target labels.
        checkpoint_path (str): Saving directory for the best model weights.
        seed (int): Random seed for reproducibility.

    Returns:
        int: The epoch number corresponding to the best validation accuracy.
    """

    # Split into train (80%) and test (20%)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    logger.info(f"Training samples: {X_train.shape[0]}, Features per sample: {X_train.shape[1]}")
    logger.info(f"Validation samples: {X_test.shape[0]}")

    weights = class_weight.compute_class_weight(
        class_weight='balanced', 
        classes=np.unique(y_train), y=y_train
    )
    class_weights = dict(enumerate(weights))

    model = keras.Sequential([
        layers.Input(shape=(X.shape[-1],)),
        layers.Dense(4, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0002),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    checkpoint = callbacks.ModelCheckpoint(
        checkpoint_path,
        monitor='val_loss',
        save_best_only=True,
        mode='min',
        save_weights_only=True,
        verbose=0
    )
    
    csv_logger = callbacks.CSVLogger("training_log.csv", append=False)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=1,
        class_weight=class_weights,
        callbacks=[checkpoint, csv_logger],
        verbose=0
    )

    best_epoch = int(np.argmin(history.history['val_loss']) + 1)

    # Load best weights before test
    model.load_weights(checkpoint_path)
    y_pred = model.predict(X_test)
    y_pred = (y_pred > 0.5).astype(int).flatten()

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    logger.info(f"Validation accuracy: {accuracy:.4f}")
    logger.info(f"Validation precision: {precision:.4f}")
    logger.info(f"Validation recall: {recall:.4f}")
    logger.info(f"Validation F1 score: {f1:.4f}")
    logger.info(f"Best epoch based on validation: {best_epoch}")

    return best_epoch


def train_final_model(X: np.ndarray, y: np.ndarray, best_epoch: int,
                    tflite_path: str = "model.tflite") -> keras.Model:
    """
    Train the final model using the best number of epochs and optionally save it as a TFLite file.

    Args:
        X (np.ndarray): Attribute matrix for training.
        y (np.ndarray): Target labels.
        best_epoch (int): Optimal number of epochs to train.
        tflite_path (str): Saving directory for the tflite model.

    Returns:
        keras.Model.
    """
    logger.info(f"Starting final training for {best_epoch} epochs...")

    weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y),
        y=y
    )
    class_weights = dict(enumerate(weights))

    model = keras.Sequential([
        layers.Input(shape=(X.shape[-1],)),
        layers.Dense(4, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer='adam', 
        loss='binary_crossentropy', 
        metrics=['accuracy']
    )
    
    csv_logger = callbacks.CSVLogger("final_training_log.csv", append=False)

    model.fit(
        X, y,
        epochs=best_epoch,
        batch_size=1,
        class_weight=class_weights,
        callbacks=[csv_logger],
        verbose=0
    )

    logger.info("Final training completed. Converting model to TFLite format...")

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    os.makedirs(os.path.dirname(tflite_path), exist_ok=True)

    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    logger.info(f"TFLite model saved to: {tflite_path}")
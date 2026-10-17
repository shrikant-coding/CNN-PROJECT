
import os
import json
import csv
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.datasets import cifar100

# Project folders
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "cifar100_cnn.keras"
METADATA_PATH = MODEL_DIR / "metadata.json"
HISTORY_PATH = BASE_DIR / "training_history.csv"

# Reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

BATCH_SIZE = 128
EPOCHS = 20
IMAGE_SIZE = 32
NUM_CLASSES = 100

CLASS_NAMES = [
    "apple", "aquarium_fish", "baby", "bear", "beaver",
    "bed", "bee", "beetle", "bicycle", "bottle",
    "bowl", "boy", "bridge", "bus", "butterfly",
    "camel", "can", "castle", "caterpillar", "cattle",
    "chair", "chimpanzee", "clock", "cloud", "cockroach",
    "couch", "crab", "crocodile", "cup", "dinosaur",
    "dolphin", "elephant", "flatfish", "forest", "fox",
    "girl", "hamster", "house", "kangaroo", "keyboard",
    "lamp", "lawn_mower", "leopard", "lion", "lizard",
    "lobster", "man", "maple_tree", "motorcycle", "mountain",
    "mouse", "mushroom", "oak_tree", "orange", "orchid",
    "otter", "palm_tree", "pear", "pickup_truck", "pine_tree",
    "plain", "plate", "poppy", "porcupine", "possum",
    "rabbit", "raccoon", "ray", "road", "rocket",
    "rose", "sea", "seal", "shark", "shrew",
    "skunk", "skyscraper", "snail", "snake", "spider",
    "squirrel", "streetcar", "sunflower", "sweet_pepper", "table",
    "tank", "telephone", "television", "tiger", "tractor",
    "train", "trout", "tulip", "turtle", "wardrobe",
    "whale", "willow_tree", "wolf", "woman", "worm"
]


def load_dataset():
    print("Loading CIFAR-100 dataset...", flush=True)

    # 50,000 training images and 10,000 test images
    (x_train, y_train), (x_test, y_test) = cifar100.load_data(
        label_mode="fine"
    )

    # Convert pixel values from 0-255 to 0-1
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    y_train = y_train.reshape(-1)
    y_test = y_test.reshape(-1)

    print(f"Training images: {len(x_train)}", flush=True)
    print(f"Testing images: {len(x_test)}", flush=True)
    print(f"Number of classes: {NUM_CLASSES}", flush=True)

    return x_train, y_train, x_test, y_test


def build_model():
    inputs = layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))

    # Data augmentation
    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomRotation(0.08)(x)
    x = layers.RandomZoom(0.1)(x)

    # CNN block 1
    x = layers.Conv2D(
        32, 3, padding="same", activation="relu"
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(
        32, 3, padding="same", activation="relu"
    )(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.20)(x)

    # CNN block 2
    x = layers.Conv2D(
        64, 3, padding="same", activation="relu"
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(
        64, 3, padding="same", activation="relu"
    )(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.25)(x)

    # CNN block 3
    x = layers.Conv2D(
        128, 3, padding="same", activation="relu"
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.30)(x)

    # Classification layers
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.40)(x)

    outputs = layers.Dense(
        NUM_CLASSES, activation="softmax"
    )(x)

    model = models.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def main():
    start_time = time.time()

    print("=" * 55, flush=True)
    print("CIFAR-100 CNN TRAINING STARTED", flush=True)
    print("=" * 55, flush=True)
    print("TensorFlow version:", tf.__version__, flush=True)

    x_train, y_train, x_test, y_test = load_dataset()

    model = build_model()
    model.summary()

    checkpoint = callbacks.ModelCheckpoint(
        filepath=str(MODEL_PATH),
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    )

    early_stop = callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=5,
        restore_best_weights=True,
        verbose=1
    )

    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=0.00001,
        verbose=1
    )

    print("\nStarting model training...", flush=True)

    history = model.fit(
        x_train,
        y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_split=0.1,
        callbacks=[checkpoint, early_stop, reduce_lr],
        verbose=1
    )

    # Evaluate the best saved model
    print("\nEvaluating on the 10,000 test images...", flush=True)

    best_model = tf.keras.models.load_model(str(MODEL_PATH))
    test_loss, test_accuracy = best_model.evaluate(
        x_test, y_test, batch_size=BATCH_SIZE, verbose=1
    )

    # Save training history
    with open(HISTORY_PATH, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["epoch", "accuracy", "val_accuracy",
                         "loss", "val_loss"])

        for i in range(len(history.history["loss"])):
            writer.writerow([
                i + 1,
                history.history["accuracy"][i],
                history.history["val_accuracy"][i],
                history.history["loss"][i],
                history.history["val_loss"][i]
            ])

    metadata = {
        "dataset": "CIFAR-100",
        "training_images": 50000,
        "test_images": 10000,
        "num_classes": NUM_CLASSES,
        "image_size": IMAGE_SIZE,
        "class_names": CLASS_NAMES,
        "test_accuracy": float(test_accuracy),
        "test_loss": float(test_loss),
        "epochs_completed": len(history.history["loss"])
    }

    with open(METADATA_PATH, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4)

    print("\n" + "=" * 55, flush=True)
    print("TRAINING COMPLETED", flush=True)
    print(f"Test accuracy: {test_accuracy * 100:.2f}%", flush=True)
    print(f"Test loss: {test_loss:.4f}", flush=True)
    print(f"Model saved to: {MODEL_PATH}", flush=True)
    print(f"History saved to: {HISTORY_PATH}", flush=True)
    print(f"Time taken: {(time.time() - start_time) / 60:.2f} minutes",
          flush=True)
    print("=" * 55, flush=True)


if __name__ == "__main__":
    main()

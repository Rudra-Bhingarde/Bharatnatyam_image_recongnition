import tensorflow as tf
import pandas as pd
import numpy as np
import os
import sys

# --- 1. Define Paths & Constants ---
test_path = "data_set/test"
model_path = 'model/model5.h5'
IMG_HEIGHT = 128
IMG_WIDTH = 128
BATCH_SIZE = 32  # <-- Using a larger batch size for speed

# --- 2. Load the Test Dataset ---
print(f"Loading test data from: {test_path}")
try:
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_path,
        batch_size=BATCH_SIZE,  # <-- Set to 32
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        shuffle=False  # No need to shuffle for evaluation
    )
except FileNotFoundError:
    print(f"Error: Test data directory not found at {test_path}")
    sys.exit(1)

# --- 3. Optimize the Data Pipeline ---
print("Optimizing data pipeline...")
AUTOTUNE = tf.data.AUTOTUNE
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

# --- 4. Load the Model ---
print(f"Loading model from {model_path}...")
try:
    model = tf.keras.models.load_model(model_path)
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit(1)

# --- 5. Evaluate the Model ---
print("Evaluating model...")
test_loss, test_acc = model.evaluate(test_ds)

print("\n--- Evaluation Results ---")
print(f'\nTest accuracy: {test_acc:.4f}')
print(f'Test loss: {test_loss:.4f}')
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import json
import random

dataset_path = "data_set/test"

img_height = 128
img_width = 128
batch_size = 32

seed = random.randint(0,100000)
train_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=(img_height,img_width),
    batch_size = batch_size
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=(img_height,img_width),
    batch_size = batch_size
)

json_filepath = 'model/class_names.json'
os.makedirs('model', exist_ok=True)
class_names = train_ds.class_names
print(f"class_names: {class_names}")
num_classes = len(class_names)
print(f"number of classes: {num_classes}")

try:
    with open(json_filepath,'w') as f:
        json.dump(class_names,f)
    print(f"Successfully saved class names to {json_filepath}")
except Exception as e:
    print(f"Error saving class names to JSON: {e}")

print(f"--- Dataset Splits ---")
print(f"Training batches:   {len(train_ds)}")
print(f"Validation/Test batches: {len(val_ds)}")
train_ds = train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

data_augmentation = tf.keras.models.Sequential([
    # Input Rescaling (standard normalization)
    tf.keras.layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
    # tf.keras.layers.RandomRotation(0.5),
    # tf.keras.layers.RandomZoom(0.1),     # Increased zoom range
    # tf.keras.layers.RandomTranslation(0.1, 0.1), # Increased translation
    # tf.keras.layers.RandomContrast(0.1), # Increased contrast
    # tf.keras.layers.RandomBrightness(0.1)

    
], name='data_preprocessing_and_augmentation')


model = tf.keras.models.Sequential([
    
    # --- Input and Augmentation Pipeline ---
    data_augmentation,

    # --- Feature Extraction Blocks (Deep CNN) ---
    
    # Block 1: 128x128 -> 64x64
    tf.keras.layers.Conv2D(32, (3, 3),padding='same'),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.MaxPool2D((2, 2)),

    # Block 2: 64x64 -> 32x32
    tf.keras.layers.Conv2D(64, (3, 3),padding='same'),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.MaxPool2D((2, 2)),

    # Block 3: 32x32 -> 16x16
    tf.keras.layers.Conv2D(128, (3, 3),padding='same'),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.MaxPool2D((2, 2)),
    # --- Classification Head ---
    
    # Parameter Reduction-
    tf.keras.layers.Flatten(),
    # Hidden Dense Layer (for final feature combination)
     # High dropout for final regularization
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.50),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.50),
    tf.keras.layers.Dense(num_classes, activation='sigmoid')
])
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00005),loss=tf.keras.losses.SparseCategoricalCrossentropy(),metrics=['accuracy'])

epochs=50


print('\n--- Starting Model Training ---')
history = model.fit(
    train_ds,
    validation_data = val_ds,
    epochs =epochs
)
model.summary()
os.makedirs('model', exist_ok=True) 
model.save('model/model6.h5')

print("--- Model Training Finished ---")
print("\n--- Evaluating on Test/Validation Data ---")
test_loss, test_acc = model.evaluate(val_ds)
print(f'\nTest accuracy: {test_acc:.4f}')
print(f'Test loss: {test_loss:.4f}')
# --- 8. Visualize Training History ---
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(epochs)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.suptitle('Model Training History')
plt.show()
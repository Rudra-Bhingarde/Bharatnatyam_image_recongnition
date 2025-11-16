import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import json

dataset_path = "data_set\Bharatanatyam-Mudra-Dataset-master"

img_height = 128
img_width = 128
batch_size = 32

train_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height,img_width),
    batch_size = batch_size
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height,img_width),
    batch_size = batch_size
)

json_filepath = 'models/class_names.json'
os.makedirs('models', exist_ok=True)
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

model = tf.keras.models.Sequential([
  tf.keras.layers.Rescaling(1./255,input_shape=(img_height,img_width,3)),
  tf.keras.layers.Conv2D(32,(3,3),activation='relu'),
  tf.keras.layers.MaxPool2D((2,2)),
  tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
  tf.keras.layers.MaxPool2D((2,2)),
  tf.keras.layers.Conv2D(128,(3,3),activation='relu'),
  tf.keras.layers.MaxPool2D((2,2)),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(256,activation='relu'),
  tf.keras.layers.Dropout(0.5),
  tf.keras.layers.Dense(128,activation='relu'),
  tf.keras.layers.Dropout(0.5),
  tf.keras.layers.Dense(64,activation='relu'),
  tf.keras.layers.Dropout(0.5),
  tf.keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam',loss=tf.keras.losses.SparseCategoricalCrossentropy(),metrics=['accuracy'])
model.summary()
epochs=40

print('\n--- Starting Model Training ---')
history = model.fit(
    train_ds,
    validation_data = val_ds,
    epochs =epochs
)
os.makedirs('models', exist_ok=True) 
model.save('models/model2.h5')

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
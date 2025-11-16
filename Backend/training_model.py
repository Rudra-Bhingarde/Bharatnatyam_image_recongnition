import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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

val_ds = tf.keras.util.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height,img_width),
    batch_size = batch_size
)

class_names = train_ds.class_names
print(f"class_names: {class_names}")
num_classes = len(class_names)
print(f"number of classes: {num_classes}")
print(f"--- Dataset Splits ---")
print(f"Training batches:   {len(train_ds)}")
print(f"Validation/Test batches: {len(val_ds)}")
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

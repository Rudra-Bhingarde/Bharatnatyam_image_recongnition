import tensorflow as tf
import numpy as np
import pandas as pd
import PIL
import io
import json
import sys

MODEL_PATH = 'model/model2.h5'
DATASET_PATH = "data_set/Bharatanatyam-Mudra-Dataset-master"
CLASSES_PATH = 'model/class_names.json'
IMG_HEIGHT = 128
IMG_WIDTH = 128


try:
    # Load the trained model
    MODEL = tf.keras.models.load_model(MODEL_PATH)
    
    # Load the class names from the JSON file
    with open(CLASSES_PATH, 'r') as f:
        CLASS_NAMES = json.load(f)
    
    print(f"Successfully loaded model and {len(CLASS_NAMES)} class names.")
    
except Exception as e:
    print(f"FATAL ERROR: Could not load model or class names. {e}")
    sys.exit(1)
def predict_image(image_bytes:bytes) -> tuple[str,float]:
    try:
        img = PIL.Image.open(io.BytesIO(image_bytes))
        img = img.conver('RGB')
        img = img.resize((IMG_WIDTH,IMG_HEIGHT))
        img_array = tf.keras.utils.imag_to_array(img)
        img_batch = tf.expand_dims(img.array,0)
        predictions = MODEL.predict(img_batch)
        score_array = predictions[0]
        predicted_index = np.argmax(score_array)
        predicted_class = CLASS_NAMES[predicted_index]
        confidence = 100 * np.max(score_array)

        return predicted_class, float(confidence)
    except PIL.UnidentifiedImageError:
        raise ValueError("Invalid image file: Could not identify image format.")
    except Exception as e:
        raise ValueError(f"error processing image: {e}")

import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D, Activation, MaxPool2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import plot_model

# --- Define dummy variables to make the model runnable ---
# (Replace with your actual augmentation layer)
data_augmentation = Layer(name='data_augmentation')
num_classes = 5 # (Replace with your actual number of classes)
# ---

model = Sequential([
    
    # 1. THE FIX: Add the Input layer here
    # This defines the input shape for the model.
    Input(shape=(128, 128, 3)),
    
    # --- Input and Augmentation Pipeline ---
    data_augmentation,

    # --- Feature Extraction Blocks (Deep CNN) ---
    
    # Block 1: 128x128 -> 64x64
    Conv2D(32, (3, 3), padding='same'),
    Activation('relu'),
    MaxPool2D((2, 2)),

    # Block 2: 64x64 -> 32x32
    Conv2D(64, (3, 3), padding='same'),
    Activation('relu'),
    MaxPool2D((2, 2)),

    # Block 3: 32x32 -> 16x16
    Conv2D(128, (3, 3), padding='same'),
    Activation('relu'),
    MaxPool2D((2, 2)),
    
    # --- Classification Head ---
    
    # Parameter Reduction-
    Flatten(),
    # Hidden Dense Layer (for final feature combination)
    # High dropout for final regularization
    Dense(256, activation='relu'),
    Dropout(0.50),
    Dense(64, activation='relu'),
    Dropout(0.50),
    Dense(num_classes, activation='sigmoid')
])

# --- Visualization 1: Text Summary ---
# This will now work without any arguments.
print("--- Model Summary ---")
model.summary()


# --- Visualization 2: Graphical Plot ---
# This saves a diagram of your model as an image file.
#
# You may need to install these libraries first:
# pip install pydot graphviz

print("\nAttempting to save model plot to 'model_architecture_LR.png'...")
try:
    plot_model(
        model,
        to_file="model_architecture_LR.png", # Changed filename to reflect LR layout
        show_shapes=True,
        show_layer_names=True,
        show_layer_activations=True,
        rankdir='LR' # <--- THIS IS THE KEY CHANGE for Left-to-Right layout
    )
    print("Successfully saved model plot with Left-to-Right layout!")
except ImportError as e:
    print(f"Error: Failed to create plot. {e}")
    print("Please make sure you have 'pydot' and 'graphviz' installed.")
    print("Run: pip install pydot graphviz")
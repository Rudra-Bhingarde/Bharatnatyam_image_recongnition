import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Rescaling, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt
import numpy as np
import sys

# --- 1. Define Parameters ---
# !!! UPDATE THIS PATH !!!
dataset_path = 'Bharatanatyam-Mudra-Dataset-master' 

img_height = 128
img_width = 128
batch_size = 32
validation_split = 0.2 # We will use 20% for testing/validation
AUTOTUNE = tf.data.AUTOTUNE

# --- 2. Load and Split the Data (Train / Test) ---

# We'll create two datasets: 80% for training, 20% for testing/validation
try:
    train_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_path,
        validation_split=validation_split,
        subset="training",     # This is the 80% training set
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )

    # We call this val_ds because model.fit expects 'validation_data'
    # This is your 20% "test" set
    val_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_path,
        validation_split=validation_split,
        subset="validation",   # This is the 20% validation/test set
        seed=123,              # Must use the same seed
        image_size=(img_height, img_width),
        batch_size=batch_size
    )
except FileNotFoundError:
    print(f"Error: The directory '{dataset_path}' was not found.")
    print("Please make sure the dataset_path variable is set correctly.")
    sys.exit()

# Get class names and number of classes
class_names = train_ds.class_names
num_classes = len(class_names)
print(f"Found {num_classes} classes: {class_names}")

print(f"--- Dataset Splits ---")
print(f"Training batches:   {len(train_ds)}")
print(f"Validation/Test batches: {len(val_ds)}")


# --- 3. Configure Data Pipeline for Performance ---
# We just cache and prefetch. The model handles normalization.
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# --- 4. Build the CNN Model ---


#[Image of a Convolutional Neural Network architecture]


model = Sequential([
    # The normalization layer is INSIDE the model
    Rescaling(1./255, input_shape=(img_height, img_width, 3)),
    
    # First Convolutional Block
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    
    # Second Convolutional Block
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    
    # Third Convolutional Block
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    
    # Flatten the results to feed into a DNN
    Flatten(),
    
    # Fully Connected Classifier Head
    Dense(128, activation='relu'),
    Dropout(0.5), # Dropout for regularization
    Dense(num_classes, activation='softmax') # Output layer
])

# --- 5. Compile the Model ---
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

# Print a summary of the model
model.summary()

# --- 6. Train the Model ---
epochs = 15 # You can increase this if needed

print("\n--- Starting Model Training ---")
history = model.fit(
    train_ds,
    validation_data=val_ds, # Using our "test" set for validation
    epochs=epochs
)
print("--- Model Training Finished ---")

# --- 7. Evaluate on the Test Set ---
# We evaluate on the same set we used for validation
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

# --- 9. Show Predictions on Test Images ---
print("\n--- Running Predictions on Test Images ---")
plt.figure(figsize=(10, 10))
# Get one batch from the test dataset
for images, labels in val_ds.take(1): # We take from val_ds
    # Make predictions on the whole batch
    predictions = model.predict(images)
    
    for i in range(9): # Show 9 images
        ax = plt.subplot(3, 3, i + 1)
        
        # Get the original image (0-255)
        img = images[i].numpy().astype("uint8") 
        
        # Get the true label
        true_label = class_names[labels[i]]
        
        # Get the predicted label
        predicted_class_index = np.argmax(predictions[i])
        predicted_label = class_names[predicted_class_index]
        confidence = 100 * np.max(predictions[i])
        
        plt.imshow(img)
        
        # Set title color
        color = "blue" if predicted_label == true_label else "red"
        
        plt.title(
            f"True: {true_label}\n"
            f"Pred: {predicted_label} ({confidence:.0f}%)",
            color=color
        )
        plt.axis("off")

plt.suptitle("Model Predictions")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
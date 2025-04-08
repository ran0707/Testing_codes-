import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from model import UNet  # Import your UNet model
from data import load_data  # Function to load your dataset

# Load dataset
train_images, train_labels, val_images, val_labels = load_data()

# Set up model
model = UNet(input_shape=(height, width, channels))  # Define your input shape
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Training parameters
epochs = 10
batch_size = 32

# Create directories for saving models and logs
os.makedirs('models', exist_ok=True)
os.makedirs('logs', exist_ok=True)

# Lists to store metrics for later visualization
train_losses = []
val_losses = []
val_accuracies = []

# Training loop
for epoch in range(epochs):
    # Train the model
    history = model.fit(train_images, train_labels, 
                        validation_data=(val_images, val_labels),
                        batch_size=batch_size, epochs=1, verbose=1)

    # Extract metrics
    train_loss = history.history['loss'][0]
    val_loss = history.history['val_loss'][0]
    val_accuracy = history.history['val_accuracy'][0]

    # Store metrics for later
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)

    print(f'Epoch {epoch + 1}/{epochs}')
    print(f'Train Loss: {train_loss:.4f}')
    print(f'Validation Loss: {val_loss:.4f}')
    print(f'Validation Accuracy: {val_accuracy:.4f}\n')

# Save model after training
model.save(os.path.join('models', 'model.h5'))

# Save metrics to a text file for later analysis
with open('logs/metrics.txt', 'w') as f:
    for i in range(epochs):
        f.write(f'Epoch {i+1}: Train Loss: {train_losses[i]:.4f}, '
                f'Validation Loss: {val_losses[i]:.4f}, '
                f'Validation Accuracy: {val_accuracies[i]:.4f}\n')

print("Training complete.")

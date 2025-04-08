import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import cv2
import random
import os

# For Grad-CAM using torchcam
from torchcam.methods import GradCAM
from torchcam.utils import overlay_mask
from torchvision.transforms.functional import normalize
from PIL import Image

# For saving to ONNX and converting to TFLite
import onnx
import tf2onnx

# Memory optimization
torch.cuda.empty_cache()

# Visualization Functions
def plot_training_history(train_losses, val_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.savefig('training_history.png')
    plt.close()

def plot_confusion_matrix(model, val_loader, device, class_names):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

def visualize_predictions(model, val_loader, device, class_names, cam_extractor, num_samples=5):
    model.eval()
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5*num_samples))
    
    for i in range(num_samples):
        inputs, labels = next(iter(val_loader))
        idx = random.randint(0, inputs.size(0)-1)
        input_image = inputs[idx].to(device)
        label = labels[idx].item()
        
        with torch.no_grad():
            output = model(input_image.unsqueeze(0))
            _, pred = torch.max(output, 1)
            pred = pred.item()
        
        # Get CAM
        activation_map = cam_extractor(output.squeeze(0).argmax().item(), output)
        heatmap = overlay_mask(np.uint8(255 * (input_image.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)),
                               activation_map.cpu().numpy(),
                               alpha=0.5)
        
        # Prepare images for plotting
        input_np = input_image.cpu().permute(1, 2, 0).numpy()
        input_np = (input_np - input_np.min()) / (input_np.max() - input_np.min())
        
        axes[i, 0].imshow(input_np)
        axes[i, 0].set_title(f'Input Image\nTrue: {class_names[label]}')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(heatmap)
        axes[i, 1].set_title(f'Predicted: {class_names[pred]}')
        axes[i, 1].axis('off')
        
        diff = np.abs(input_np - heatmap)
        axes[i, 2].imshow(diff)
        axes[i, 2].set_title('Difference Map')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('prediction_visualization.png')
    plt.close()

def generate_cam_visualization(model, val_loader, device, class_names, cam_extractor, num_samples=5):
    model.eval()
    fig, axes = plt.subplots(num_samples, 2, figsize=(10, 5*num_samples))
    
    for i in range(num_samples):
        inputs, labels = next(iter(val_loader))
        idx = random.randint(0, inputs.size(0)-1)
        input_image = inputs[idx].to(device)
        label = labels[idx].item()
        
        with torch.no_grad():
            output = model(input_image.unsqueeze(0))
            pred_class = output.argmax(dim=1).item()
        
        # Get CAM
        activation_map = cam_extractor(pred_class, output)
        heatmap = overlay_mask(np.uint8(255 * (input_image.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)),
                               activation_map.cpu().numpy(),
                               alpha=0.5)
        
        # Prepare images for plotting
        input_np = input_image.cpu().permute(1, 2, 0).numpy()
        input_np = (input_np - input_np.min()) / (input_np.max() - input_np.min())
        
        axes[i, 0].imshow(input_np)
        axes[i, 0].set_title(f'Original Image\nTrue: {class_names[label]}')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(heatmap)
        axes[i, 1].set_title('Class Activation Map')
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('cam_visualization.png')
    plt.close()

def generate_all_visualizations(model, train_losses, val_losses, val_loader, device, class_names, cam_extractor):
    print("Generating training history plot...")
    plot_training_history(train_losses, val_losses)
    
    print("Generating confusion matrix and classification report...")
    plot_confusion_matrix(model, val_loader, device, class_names)
    
    print("Generating prediction visualizations...")
    visualize_predictions(model, val_loader, device, class_names, cam_extractor)
    
    print("Generating CAM visualizations...")
    generate_cam_visualization(model, val_loader, device, class_names, cam_extractor)

# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=25, accumulation_steps=4):
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()
        
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels) / accumulation_steps
            loss.backward()
            
            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            train_loss += loss.item() * inputs.size(0) * accumulation_steps
        
        if (i + 1) % accumulation_steps != 0:
            optimizer.step()
            optimizer.zero_grad()
            
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        if epoch % 5 == 0:
            torch.cuda.empty_cache()

    # Initialize Grad-CAM extractor
    cam_extractor = GradCAM(model, target_layer='layer4')  # ResNet-18's last convolutional layer
    # Generate visualizations after training
    generate_all_visualizations(model, train_losses, val_losses, val_loader, device, train_loader.dataset.dataset.classes, cam_extractor)
    
    return train_losses, val_losses

# Data loading and preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ResNet expects 224x224
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Standard normalization for ImageNet
                         std=[0.229, 0.224, 0.225]),
])

# Load dataset
dataset = ImageFolder(root='/home/idrone2/Tea_pest/Tea-TJ', transform=transform)
class_names = dataset.classes
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

# Initialize model, criterion, and optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(class_names))  # Adjust the final layer for your number of classes
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
train_losses, val_losses = train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=25)

# Save the model in PyTorch format
torch.save(model.state_dict(), 'resnet18_model.pth')
print("Model saved as resnet18_model.pth")

# Save the model in ONNX format
dummy_input = torch.randn(1, 3, 224, 224, device=device)
onnx_model_path = "resnet18_model.onnx"
torch.onnx.export(model, dummy_input, onnx_model_path, 
                  input_names=['input'], output_names=['output'], 
                  dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
                  opset_version=11)
print(f"Model exported to ONNX format at {onnx_model_path}")

# Convert ONNX model to TensorFlow
import tf2onnx
import subprocess

# First, install the required packages if not already installed
# pip install onnx tf2onnx

# Convert ONNX to TensorFlow
tf_model_path = "resnet18_model_tf"
if not os.path.exists(tf_model_path):
    os.makedirs(tf_model_path)

command = f"python -m tf2onnx.convert --onnx {onnx_model_path} --output {tf_model_path}/model.pb --inputs input:0 --outputs output:0"
subprocess.run(command, shell=True)
print(f"Model converted to TensorFlow format at {tf_model_path}/model.pb")

# Convert TensorFlow model to TFLite
import tensorflow as tf

# Load the TensorFlow model
converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_path)
tflite_model = converter.convert()

# Save the TFLite model
with open("resnet18_model.tflite", "wb") as f:
    f.write(tflite_model)
print("Model converted to TFLite format at resnet18_model.tflite")

# (Optional) Convert to H5 using Keras
# Note: Direct conversion from PyTorch to H5 is not straightforward. It's recommended to use PyTorch's `.pth` or ONNX formats.
# If you specifically need an H5 file, consider rebuilding the model architecture in Keras and transferring the weights manually.

print("All tasks completed successfully.")

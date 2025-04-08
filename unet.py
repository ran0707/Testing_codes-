
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
from tqdm import tqdm



torch.cuda.empty_cache()

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)
    
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()

        self.enc1 = DoubleConv(in_channels, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)
        
        self.pool = nn.MaxPool2d(2)
        
        self.dec4 = DoubleConv(512 + 256, 256)
        self.dec3 = DoubleConv(256 + 128, 128)
        self.dec2 = DoubleConv(128 + 64, 64)
        self.dec1 = nn.Conv2d(64, out_channels, 1)

    def forward(self, x): 
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Decoder
        d4 = self.dec4(torch.cat([nn.functional.interpolate(e4, scale_factor=2, mode='bilinear', align_corners=True), e3], dim=1))
        d3 = self.dec3(torch.cat([nn.functional.interpolate(d4, scale_factor=2, mode='bilinear', align_corners=True), e2], dim=1))
        d2 = self.dec2(torch.cat([nn.functional.interpolate(d3, scale_factor=2, mode='bilinear', align_corners=True), e1], dim=1))
        
        return self.dec1(d2)

# Data loading and preprocessing
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])


dataset = ImageFolder(root='/home/idrone2/Tea_pest/Tea-TJ', transform=transform)


train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)


# Model, loss, and optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet(in_channels=3, out_channels=3).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# Training loop
scaler = torch.cuda.amp.GradScaler()

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    train_losses = []
    val_train_losses = []
    val_losses = []
    val_val_losses = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print('-' * 20)

        # Progress bar for the training loop
        train_loader_tqdm = tqdm(train_loader, desc="Training", leave=False)

        for inputs, _ in train_loader_tqdm:
            inputs = inputs.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, inputs)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            # Update the progress bar description with the current batch loss
            train_loader_tqdm.set_postfix({"Batch Loss": loss.item()})

        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        print(f"Train Loss: {train_loss:.4f}")

        # Validation phase (on training data without updating weights)
        model.eval()
        val_train_loss = 0.0
        val_loss = 0.0

        # Progress bar for the validation training loop
        val_train_loader_tqdm = tqdm(train_loader, desc="Validation Train", leave=False)
        val_loader_tqdm = tqdm(val_loader, desc="Validation Val", leave=False)

        with torch.no_grad():
            for inputs, _ in val_train_loader_tqdm:
                inputs = inputs.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, inputs)
                val_train_loss += loss.item() * inputs.size(0)
                val_train_loader_tqdm.set_postfix({"Batch Loss": loss.item()})

        val_train_loss /= len(train_loader.dataset)
        val_train_losses.append(val_train_loss)

        print(f"Validation Train Loss: {val_train_loss:.4f}")

        with torch.no_grad():
            for inputs, _ in val_loader_tqdm:
                inputs = inputs.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, inputs)
                val_loss += loss.item() * inputs.size(0)
                val_loader_tqdm.set_postfix({"Batch Loss": loss.item()})

        val_loss /= len(val_loader.dataset)
        val_val_losses.append(val_loss)

        print(f"Validation Loss: {val_loss:.4f}")

    return train_losses, val_train_losses, val_val_losses, val_losses

train_losses, val_train_losses, val_val_losses, val_losses = train_model(model, train_loader, val_loader, criterion, optimizer)


# Plot the losses
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Losses')
plt.legend()
plt.show()

import numpy as np
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

import os

def plot_results(model, loader, num_images=5, save_dir="plots"):
    # Create directory to save plots if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model.eval()  # Set the model to evaluation mode
    fig, axs = plt.subplots(num_images, 3, figsize=(15, 5 * num_images))
    
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(loader):
            if i >= num_images:
                break
            inputs = inputs.to(device)
            outputs = model(inputs)
            
            # Original Image
            original_img = inputs[0].cpu().permute(1, 2, 0).numpy()
            
            # Ground Truth (for segmentation tasks, it's assumed to be in the labels)
            if labels is not None and len(labels) > 0:
                if labels.dim() == 3:
                    ground_truth = labels[0].cpu().numpy()
                elif labels.dim() == 4:
                    ground_truth = labels[0].cpu().permute(1, 2, 0).numpy()
                else:
                    ground_truth = None
            else:
                ground_truth = None
            
            # Predicted Output (Segmentation Map)
            predicted = torch.argmax(outputs[0], dim=0).cpu().numpy()
            
            # Plot the Original Image
            axs[i, 0].imshow(original_img)
            axs[i, 0].set_title('Original Image')
            axs[i, 0].axis('off')
            
            # Plot the Ground Truth if available
            if ground_truth is not None:
                axs[i, 1].imshow(ground_truth)
                axs[i, 1].set_title('Ground Truth')
                axs[i, 1].axis('off')
            else:
                axs[i, 1].set_title('No Ground Truth Available')
                axs[i, 1].axis('off')
            
            # Plot the Predicted Output
            axs[i, 2].imshow(predicted)
            axs[i, 2].set_title('Predicted Output')
            axs[i, 2].axis('off')
        
        # Save each plot to a file
        plot_filename = os.path.join(save_dir, f'result_plot_{i + 1}.png')
        plt.savefig(plot_filename)
        print(f"Plot saved as {plot_filename}")
    
    plt.close(fig)  # Close figure to free up memory

    model.eval()  # Set the model to evaluation mode
    fig, axs = plt.subplots(num_images, 3, figsize=(15, 5 * num_images))
    
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(loader):
            if i >= num_images:
                break
            inputs = inputs.to(device)
            outputs = model(inputs)
            
            # Original Image
            original_img = inputs[0].cpu().permute(1, 2, 0).numpy()
            
            # Ground Truth (for segmentation tasks, it's assumed to be in the labels)
            if labels is not None and len(labels) > 0:
                if labels.dim() == 3:
                    # Single-channel ground truth (height, width)
                    ground_truth = labels[0].cpu().numpy()
                elif labels.dim() == 4:
                    # If labels have a channel dimension (batch_size, channels, height, width)
                    ground_truth = labels[0].cpu().permute(1, 2, 0).numpy()
                else:
                    ground_truth = None
            else:
                ground_truth = None
            
            # Predicted Output (Segmentation Map)
            predicted = torch.argmax(outputs[0], dim=0).cpu().numpy()
            
            # Plot the Original Image
            axs[i, 0].imshow(original_img)
            axs[i, 0].set_title('Original Image')
            axs[i, 0].axis('off')
            
            # Plot the Ground Truth if available
            if ground_truth is not None:
                axs[i, 1].imshow(ground_truth)
                axs[i, 1].set_title('Ground Truth')
                axs[i, 1].axis('off')
            else:
                axs[i, 1].set_title('No Ground Truth Available')
                axs[i, 1].axis('off')
            
            # Plot the Predicted Output
            axs[i, 2].imshow(predicted)
            axs[i, 2].set_title('Predicted Output')
            axs[i, 2].axis('off')

    plt.show()

    model.eval()  # Set the model to evaluation mode
    fig, axs = plt.subplots(num_images, 3, figsize=(15, 5 * num_images))
    
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(loader):
            if i >= num_images:
                break
            inputs = inputs.to(device)
            outputs = model(inputs)
            
            # Original Image
            original_img = inputs[0].cpu().permute(1, 2, 0).numpy()
            
            # Ground Truth (for segmentation tasks, it's assumed to be in the labels)
            ground_truth = labels[0].cpu().permute(1, 2, 0).numpy() if labels is not None else None
            
            # Predicted Output (Segmentation Map)
            predicted = torch.argmax(outputs[0], dim=0).cpu().numpy()
            
            # Plot the Original Image
            axs[i, 0].imshow(original_img)
            axs[i, 0].set_title('Original Image')
            axs[i, 0].axis('off')
            
            # Plot the Ground Truth
            if ground_truth is not None:
                axs[i, 1].imshow(ground_truth)
                axs[i, 1].set_title('Ground Truth')
                axs[i, 1].axis('off')
            
            # Plot the Predicted Output
            axs[i, 2].imshow(predicted)
            axs[i, 2].set_title('Predicted Output')
            axs[i, 2].axis('off')

    plt.show()

# Example usage:
plot_results(model, val_loader)



from torchvision import models
import torch.nn.functional as F

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        
        # Hook the gradients of the target layer
        target_layer.register_backward_hook(self.save_gradient)
    
    def save_gradient(self, grad):
        self.gradients = grad

    def forward(self, x):
        return self.model(x)
    
    def get_grad_cam(self, input_image, target_class):
        # Forward pass
        model_output = self.forward(input_image)
        
        # Get the target output
        target = model_output[:, target_class]
        
        # Backward pass
        self.model.zero_grad()
        target.backward()
        
        # Get the gradients and target feature map
        gradients = self.gradients[0]
        feature_map = self.target_layer.output
        
        # Global average pooling
        weights = torch.mean(gradients, dim=(2, 3))
        
        # Compute Grad-CAM
        grad_cam = torch.zeros(feature_map.shape[2:], dtype=torch.float32)
        for i, w in enumerate(weights):
            grad_cam += w * feature_map[i, :, :]
        
        grad_cam = F.relu(grad_cam)
        grad_cam = (grad_cam - grad_cam.min()) / (grad_cam.max() - grad_cam.min())  # Normalize
        
        return grad_cam

# To use Grad-CAM, specify the target layer in the UNet (e.g., after one of the decoders)
# grad_cam = GradCAM(model, model.enc4) 

from sklearn.metrics import confusion_matrix, classification_report

def evaluate_model(model, loader, save_dir="plots"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    all_preds = []
    all_labels = []
    
    model.eval()
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    # Flatten the lists
    all_preds = np.concatenate(all_preds).flatten()
    all_labels = np.concatenate(all_labels).flatten()
    
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")

    # Save confusion matrix plot
    cm_filename = os.path.join(save_dir, 'confusion_matrix.png')
    plt.savefig(cm_filename)
    print(f"Confusion matrix saved as {cm_filename}")
    plt.close()
    
    # Classification Report
    print(classification_report(all_labels, all_preds))

    all_preds = []
    all_labels = []
    
    model.eval()
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    # Flatten the lists
    all_preds = np.concatenate(all_preds).flatten()
    all_labels = np.concatenate(all_labels).flatten()
    
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()
    
    # Classification Report
    print(classification_report(all_labels, all_preds))

# Example usage:
evaluate_model(model, val_loader)

from collections import Counter

def dataset_composition_report(dataset):
    class_counts = Counter([label for _, label in dataset])
    classes = list(class_counts.keys())
    counts = list(class_counts.values())
    
    # Plot class distribution
    plt.figure(figsize=(10, 5))
    plt.bar(classes, counts)
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.title("Dataset Class Composition")
    plt.show()

# Example usage:
dataset_composition_report(train_dataset)


import onnx
import onnx_tf
import tensorflow as tf
import h5py

def save_model_onnx(model, save_path):
    # Prepare an example input tensor
    dummy_input = torch.randn(1, 3, 256, 256).to(device)
    
    # Export the model
    torch.onnx.export(model, dummy_input, save_path, verbose=True, opset_version=12)
    print(f"Model saved in ONNX format at {save_path}")


def save_model_h5(model, save_path):
    # Convert to ONNX first
    onnx_path = "temp_model.onnx"
    save_model_onnx(model, onnx_path)
    
    # Load ONNX model and convert to TensorFlow
    onnx_model = onnx.load(onnx_path)
    tf_rep = onnx_tf.backend.prepare(onnx_model)
    
    # Save as H5
    tf_rep.export_graph(save_path)
    print(f"Model saved in H5 format at {save_path}")


def save_model_tflite(model, save_path):
    # Convert to ONNX first
    onnx_path = "temp_model.onnx"
    save_model_onnx(model, onnx_path)
    
    # Load ONNX model and convert to TensorFlow
    onnx_model = onnx.load(onnx_path)
    tf_rep = onnx_tf.backend.prepare(onnx_model)
    
    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_saved_model(tf_rep.export_graph())
    tflite_model = converter.convert()
    
    # Save TFLite model
    with open(save_path, 'wb') as f:
        f.write(tflite_model)
    print(f"Model saved in TFLite format at {save_path}")


# Usage:
# Assuming you have trained your model and it's stored in the 'model' variable
save_model_onnx(model, "unet_model.onnx")
save_model_h5(model, "unet_model.h5")
save_model_tflite(model, "unet_model.tflite")
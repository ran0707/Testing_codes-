import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, models, transforms
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import random
from PIL import Image

# Step 1: Define the transforms for your dataset
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Step 2: Load your local dataset with 3 classes
data_dir = '/home/idrone2/Desktop/rk/Alzheimer_dataset'  # Path to your dataset

# Assuming folder structure: dataset/class_name/image.jpg
image_datasets = datasets.ImageFolder(data_dir, transform=data_transforms)
class_names = image_datasets.classes  # Get class names like ['Mild_Demented', 'Non_Demented', 'Very_Mild_Demented']
print("Class names:", class_names)

# Step 3: Load the pre-trained ResNet18 model
model = models.resnet18(weights='ResNet18_Weights.DEFAULT')  # Use 'weights' instead of 'pretrained'

# Modify the final layer for 3 classes
num_classes = len(class_names)
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Set model to evaluation mode
model.eval()

# Step 4: Hook to extract feature maps from the last convolutional layer
features_blobs = []
def hook_feature(module, input, output):
    features_blobs.append(output)

# Register the hook to the last conv layer
final_conv_layer = model.layer4[1].conv2
final_conv_layer.register_forward_hook(hook_feature)

# Get weights from the fully connected layer
params = list(model.parameters())
weight_softmax = np.squeeze(params[-2].cpu().data.numpy())

# Function to generate CAM
def generate_cam(feature_conv, weight_fc, class_idx):
    bz, nc, h, w = feature_conv.shape
    cam = weight_fc[class_idx].dot(feature_conv.reshape((nc, h * w)))
    cam = cam.reshape(h, w)
    
    # Normalize to 0-1 range
    cam = cam - np.min(cam)
    cam = cam / np.max(cam)
    cam = np.uint8(255 * cam)
    
    return cam

# Function to overlay CAM on the image
def show_cam_on_image(img, cam):
    heatmap = cv2.applyColorMap(cv2.resize(cam, (img.shape[2], img.shape[1])), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    img = img.numpy().transpose(1, 2, 0)  # Convert tensor to NumPy array
    img = np.clip(img, 0, 1)
    cam_img = heatmap + img
    cam_img = cam_img / np.max(cam_img)
    return np.uint8(255 * cam_img)

# Function to load a random image from a class directory
def get_random_image_from_class(class_dir):
    images = os.listdir(class_dir)
    img_name = random.choice(images)
    img_path = os.path.join(class_dir, img_name)
    return img_path

# Function to process and visualize CAM for a single image
def process_and_visualize_cam(image_path, model, class_names):
    # Load an image
    img = Image.open(image_path).convert('RGB')
    img_tensor = data_transforms(img).unsqueeze(0)  # Transform and add batch dimension
    
    # Perform forward pass to get model output
    outputs = model(img_tensor)
    _, pred_idx = torch.max(outputs, 1)
    pred_class = class_names[pred_idx]
    
    # Extract feature maps
    features = features_blobs[0].cpu().data.numpy()

    # Generate CAM
    cam = generate_cam(features, weight_softmax, pred_idx)
    
    # Get CAM overlay on the image
    cam_img = show_cam_on_image(img_tensor.squeeze(0), cam)
    return cam_img, img_tensor.squeeze(0).numpy().transpose(1, 2, 0), pred_class  # Return CAM, original image, and predicted class name

# Step 5: Display and save all CAMs in a single composite image with class names
def display_and_save_all_cams_in_one_image(model, data_dir, class_names, output_image_path):
    cam_images = []  # List to store all CAM images
    predicted_classes = []  # List to store predicted class names for each CAM
    
    for class_name in class_names:
        class_dir = os.path.join(data_dir, class_name)
        
        # Select 2 random images from each class
        for _ in range(2):
            random_image_path = get_random_image_from_class(class_dir)
            
            # Get CAM, original image, and predicted class
            cam_img, _, pred_class = process_and_visualize_cam(random_image_path, model, class_names)
            
            # Append CAM image and class name to the lists
            cam_images.append(cam_img)
            predicted_classes.append(pred_class)
    
    # Step 6: Create a 2x3 grid composite image
    rows, cols = 2, 3  # 2 rows, 3 columns for 6 images
    cam_height, cam_width, _ = cam_images[0].shape
    composite_image = np.zeros((rows * cam_height, cols * cam_width, 3), dtype=np.uint8)
    
    # Fill the composite image with the 6 CAM images and add class names
    for idx, (cam_img, pred_class) in enumerate(zip(cam_images, predicted_classes)):
        row = idx // cols
        col = idx % cols
        composite_image[row * cam_height:(row + 1) * cam_height, col * cam_width:(col + 1) * cam_width, :] = cam_img
        
        # Add class name as text to the CAM image
        text_position = (col * cam_width + 5, row * cam_height + 25)  # Adjust position
        font_scale = 0.7  # Reduced font size
        font_thickness = 1  # Reduced thickness to make the text fit better
        cv2.putText(composite_image, pred_class, text_position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness, lineType=cv2.LINE_AA)

    # Save the composite image
    cv2.imwrite(output_image_path, cv2.cvtColor(composite_image, cv2.COLOR_RGB2BGR))
    
    # Step 7: Display the composite image
    plt.imshow(composite_image)
    plt.title('Composite CAM Image (2x3) with Class Names')
    plt.axis('off')
    plt.show()

# Step 8: Run the function to display and save CAMs in a single image
output_image_path = './composite_cam_image_2x3_with_class_names.jpg'  # Path to save the final composite image
display_and_save_all_cams_in_one_image(model, data_dir, class_names, output_image_path)

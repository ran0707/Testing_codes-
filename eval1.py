import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import os
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch.nn.functional as F

from train import UNet  # Assuming UNet is defined in train.py

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def plot_results(model, loader, num_images=5, save_dir="plots"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model.eval()
    fig, axs = plt.subplots(num_images, 3, figsize=(15, 5 * num_images))
    
    with torch.no_grad():
        for i, (inputs, _) in enumerate(loader):
            if i >= num_images:
                break
            inputs = inputs.to(device)
            outputs = model(inputs)
            
            original_img = inputs[0].cpu().permute(1, 2, 0).numpy()
            predicted = outputs[0].cpu().permute(1, 2, 0).numpy()

            axs[i, 0].imshow(original_img)
            axs[i, 0].set_title('Original Image')
            axs[i, 0].axis('off')
            
            axs[i, 1].imshow(predicted)
            axs[i, 1].set_title('Predicted Output')
            axs[i, 1].axis('off')
            
            diff = np.abs(original_img - predicted)
            axs[i, 2].imshow(diff)
            axs[i, 2].set_title('Difference')
            axs[i, 2].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'results.png'))
    plt.close()

def evaluate_model(model, loader, save_dir="plots"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    model.eval()
    total_loss = 0
    criterion = torch.nn.MSELoss()

    with torch.no_grad():
        for inputs, _ in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    print(f"Average Loss: {avg_loss:.4f}")

    with open(os.path.join(save_dir, 'evaluation_results.txt'), 'w') as f:
        f.write(f"Average Loss: {avg_loss:.4f}\n")

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_image):
        self.model.zero_grad()
        output = self.model(input_image)
        output.backward(torch.ones_like(output))

        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        for i in range(self.activations.size(1)):
            self.activations[:, i, :, :] *= pooled_gradients[i]
        
        heatmap = torch.mean(self.activations, dim=1).squeeze()
        heatmap = F.relu(heatmap)
        heatmap /= torch.max(heatmap)
        
        return heatmap.detach().cpu().numpy()

def main():
    # Load data
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    dataset_path = '/home/idrone2/Tea_pest/Tea-TJ'
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    dataset = ImageFolder(root=dataset_path, transform=transform)
    loader = DataLoader(dataset, batch_size=16, shuffle=False)

    # Load model
    model = UNet(in_channels=3, out_channels=3).to(device)
    checkpoint = torch.load('model_checkpoint.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Evaluate model
    evaluate_model(model, loader)

    # Plot results
    plot_results(model, loader, num_images=5)

    # Generate GradCAM
    grad_cam = GradCAM(model, model.enc4)
    
    for i, (input_image, _) in enumerate(loader):
        if i >= 5:  # Generate GradCAM for first 5 images
            break
        input_image = input_image.to(device)
        heatmap = grad_cam.generate(input_image)
        
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(input_image[0].cpu().permute(1, 2, 0).numpy())
        plt.title('Original Image')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(heatmap, cmap='jet')
        plt.title('GradCAM')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(f'plots/gradcam_{i}.png')
        plt.close()

if __name__ == "__main__":
    main()
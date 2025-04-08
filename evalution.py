import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import os
from torchvision import models
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def plot_results(model, loader, num_images=5, save_dir="plots"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model.eval()
    fig, axs = plt.subplots(num_images, 3, figsize=(15, 5 * num_images))
    
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(loader):
            if i >= num_images:
                break
            inputs = inputs.to(device)
            outputs = model(inputs)
            
            original_img = inputs[0].cpu().permute(1, 2, 0).numpy()
            ground_truth = labels[0].cpu().permute(1, 2, 0).numpy() if labels is not None else None
            predicted = torch.argmax(outputs[0], dim=0).cpu().numpy()

            axs[i, 0].imshow(original_img)
            axs[i, 0].set_title('Original Image')
            axs[i, 0].axis('off')

            if ground_truth is not None:
                axs[i, 1].imshow(ground_truth)
                axs[i, 1].set_title('Ground Truth')
                axs[i, 1].axis('off')
            
            axs[i, 2].imshow(predicted)
            axs[i, 2].set_title('Predicted Output')
            axs[i, 2].axis('off')

    plt.show()

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

    all_preds = np.concatenate(all_preds).flatten()
    all_labels = np.concatenate(all_labels).flatten()

    cm = confusion_matrix(all_labels, all_preds)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
    plt.close()

    print(classification_report(all_labels, all_preds))

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        target_layer.register_backward_hook(self.save_gradient)

    def save_gradient(self, grad):
        self.gradients = grad

    def forward(self, x):
        return self.model(x)
    
    def get_grad_cam(self, input_image, target_class):
        model_output = self.forward(input_image)
        target = model_output[:, target_class]
        self.model.zero_grad()
        target.backward()

        gradients = self.gradients[0]
        feature_map = self.target_layer.output
        weights = torch.mean(gradients, dim=[2, 3])

        cam = torch.zeros(feature_map.shape[2:], dtype=torch.float32)
        for i, w in enumerate(weights):
            cam += w * feature_map[0, i, :, :]

        cam = F.relu(cam)
        cam = (cam - cam.min()) / (cam.max() - cam.min())  # Normalize the heatmap
        return cam.detach().cpu().numpy()

model = UNet(in_channels=3, out_channels=3).to(device)
checkpoint = torch.load('model_checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# Load data and evaluate
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

dataset = ImageFolder(root='/home/idrone2/Tea_pest/Tea-TJ', transform=transform)
loader = DataLoader(dataset, batch_size=16, shuffle=False)

evaluate_model(model, loader)
plot_results(model, loader, num_images=5)

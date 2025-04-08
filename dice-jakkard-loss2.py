import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
from datetime import datetime

# Dataset class to load images
class CustomDataset(Dataset):
    """Custom dataset to load images and assign labels based on folder structure"""
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = os.listdir(root_dir)  # Class folder names
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}  # Map class names to indices
        self.image_paths = []
        self.labels = []

        # Collect all image paths and corresponding labels
        for cls_name in self.classes:
            cls_folder = os.path.join(root_dir, cls_name)
            cls_label = self.class_to_idx[cls_name]
            for img_name in os.listdir(cls_folder):
                img_path = os.path.join(cls_folder, img_name)
                self.image_paths.append(img_path)
                self.labels.append(cls_label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

# Define transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # You can adjust size if needed
    transforms.ToTensor(),
])

# Dataset directory structure (Replace this with your dataset path)
dataset_dir = '/home/idrone2/Desktop/Kaggle_datasets/alzheimer'  # Update to your dataset path
train_dataset = CustomDataset(root_dir=dataset_dir, transform=transform)

# Split dataset into train/validation (80% train, 20% validation)
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Metrics Tracker Class
class MetricsTracker:
    def __init__(self, results_dir):
        self.results_dir = results_dir
        self.metrics = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'train_dice': [],
            'val_dice': [],
            'train_jaccard': [],
            'val_jaccard': []
        }
        
        # Create results directory if it doesn't exist
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
    
    def update(self, new_metrics):
        """Update metrics with the new values for the current epoch."""
        for key, value in new_metrics.items():
            if key in self.metrics:
                self.metrics[key].append(value)
    
    def plot_metrics(self):
        """Plot and save the metrics as images."""
        for metric_name in self.metrics:
            plt.figure()
            plt.plot(self.metrics[metric_name], label=metric_name)
            plt.title(f'{metric_name} over epochs')
            plt.xlabel('Epoch')
            plt.ylabel(metric_name)
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(self.results_dir, f'{metric_name}.png'))
            plt.close()
    
    def save_metrics(self):
        """Save metrics to a JSON file."""
        metrics_file = os.path.join(self.results_dir, 'metrics.json')
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=4)
    
    def generate_statistics(self):
        """Generate final statistics (mean, min, max) for each metric."""
        stats = {}
        for key, values in self.metrics.items():
            stats[key] = {
                'mean': sum(values) / len(values) if values else 0,
                'min': min(values) if values else 0,
                'max': max(values) if values else 0
            }
        return stats

# Dice and Jaccard coefficient functions (dummy implementations for now)
def dice_coefficient(output, target):
    """Calculates Dice coefficient (dummy implementation for classification)."""
    return (2 * (output.argmax(1) == target).sum().item()) / (output.size(0) + target.size(0))

def jaccard_coefficient(output, target):
    """Calculates Jaccard index (dummy implementation for classification)."""
    return (output.argmax(1) == target).sum().item() / target.size(0)

# EnhancedTrainer Class
class EnhancedTrainer:
    def __init__(self, model, train_loader, val_loader, device, 
                 learning_rate=0.001, num_epochs=100):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.num_epochs = num_epochs
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5)
        
        # Create results directory with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.results_dir = f'training_results_{timestamp}'
        self.metrics_tracker = MetricsTracker(self.results_dir)
    
    def train_epoch(self):
        self.model.train()
        metrics = {
            'loss': 0.0,
            'acc': 0.0,
            'dice': 0.0,
            'jaccard': 0.0
        }
        total = 0
        
        progress_bar = tqdm(self.train_loader, desc="Training", unit="batch")
        
        for data, target in progress_bar:
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            metrics['loss'] += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            metrics['acc'] += predicted.eq(target).sum().item()
            metrics['dice'] += dice_coefficient(output, target)
            metrics['jaccard'] += jaccard_coefficient(output, target)
            
            progress_bar.set_postfix({
                'Loss': f"{metrics['loss'] / (len(progress_bar)):.4f}",
                'Acc': f"{100. * metrics['acc'] / total:.2f}%",
                'Dice': f"{metrics['dice'] / len(progress_bar):.4f}",
                'Jaccard': f"{metrics['jaccard'] / len(progress_bar):.4f}"
            })
        
        num_batches = len(self.train_loader)
        return {
            'loss': metrics['loss'] / num_batches,
            'acc': 100. * metrics['acc'] / total,
            'dice': metrics['dice'] / num_batches,
            'jaccard': metrics['jaccard'] / num_batches
        }
    
    def validate(self):
        self.model.eval()
        metrics = {
            'loss': 0.0,
            'acc': 0.0,
            'dice': 0.0,
            'jaccard': 0.0
        }
        total = 0
        
        progress_bar = tqdm(self.val_loader, desc="Validation", unit="batch")
        
        with torch.no_grad():
            for data, target in progress_bar:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                
                loss = self.criterion(output, target)
                metrics['loss'] += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                metrics['acc'] += predicted.eq(target).sum().item()
                metrics['dice'] += dice_coefficient(output, target)
                metrics['jaccard'] += jaccard_coefficient(output, target)
                
                progress_bar.set_postfix({
                    'Loss': f"{metrics['loss'] / (len(progress_bar)):.4f}",
                    'Acc': f"{100. * metrics['acc'] / total:.2f}%",
                    'Dice': f"{metrics['dice'] / len(progress_bar):.4f}",
                    'Jaccard': f"{metrics['jaccard'] / len(progress_bar):.4f}"
                })
        
        num_batches = len(self.val_loader)
        return {
            'loss': metrics['loss'] / num_batches,
            'acc': 100. * metrics['acc'] / total,
            'dice': metrics['dice'] / num_batches,
            'jaccard': metrics['jaccard'] / num_batches
        }
    
    def train(self):
        best_val_acc = 0.0
        
        for epoch in range(self.num_epochs):
            print(f'\nEpoch {epoch+1}/{self.num_epochs}')
            
            train_metrics = self.train_epoch()
            print(f"Training - Loss: {train_metrics['loss']:.4f}, "
                  f"Acc: {train_metrics['acc']:.2f}%, "
                  f"Dice: {train_metrics['dice']:.4f}, "
                  f"Jaccard: {train_metrics['jaccard']:.4f}")
            
            val_metrics = self.validate()
            print(f"Validation - Loss: {val_metrics['loss']:.4f}, "
                  f"Acc: {val_metrics['acc']:.2f}%, "
                  f"Dice: {val_metrics['dice']:.4f}, "
                  f"Jaccard: {val_metrics['jaccard']:.4f}")
            
            # Update learning rate based on validation loss
            self.scheduler.step(val_metrics['loss'])
            
            # Track metrics
            self.metrics_tracker.update({
                'train_loss': train_metrics['loss'],
                'val_loss': val_metrics['loss'],
                'train_acc': train_metrics['acc'],
                'val_acc': val_metrics['acc'],
                'train_dice': train_metrics['dice'],
                'val_dice': val_metrics['dice'],
                'train_jaccard': train_metrics['jaccard'],
                'val_jaccard': val_metrics['jaccard'],
            })
            
            # Save best model
            if val_metrics['acc'] > best_val_acc:
                best_val_acc = val_metrics['acc']
                torch.save(self.model.state_dict(), os.path.join(self.results_dir, 'best_model.pth'))
        
        # Plot and save metrics at the end of training
        self.metrics_tracker.plot_metrics()
        self.metrics_tracker.save_metrics()
        
        # Return final stats
        return self.metrics_tracker.generate_statistics()

# Example CNN model for training
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=3):  # Adjust num_classes based on your dataset
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 64 * 64, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Main training loop
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = SimpleCNN(num_classes=3).to(device)  # Adjust num_classes as per your dataset
    
    trainer = EnhancedTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=0.001,
        num_epochs=100
    )
    
    # Start training
    final_stats = trainer.train()
    
    # Print final statistics
    print("Final Statistics:")
    for key, value in final_stats.items():
        print(f"{key}: mean={value['mean']:.4f}, min={value['min']:.4f}, max={value['max']:.4f}")

if __name__ == "__main__":
    main()


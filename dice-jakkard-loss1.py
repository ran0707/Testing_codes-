import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json
from pathlib import Path

def create_data_loaders(data_dir, batch_size=32, train_split=0.8, num_workers=2):
    """Create train and validation data loaders"""
    
    # Create dataset
    full_dataset = ImageClassificationDataset(data_dir)
    
    # Calculate lengths for train and validation splits
    total_size = len(full_dataset)
    train_size = int(train_split * total_size)
    val_size = total_size - train_size
    
    # Split dataset
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_loader, val_loader

class ImageClassificationDataset(Dataset):
    """Dataset class for loading images for 3-class classification"""
    
    def __init__(self, data_dir, transform=None):
        """
        Args:
            data_dir (str): Directory path containing class folders
            transform: Optional transform to be applied to images
        """
        self.data_dir = Path(data_dir)
        self.transform = transform if transform else self.get_default_transforms()
        
        # Get all class folders
        self.classes = sorted([d for d in os.listdir(data_dir) 
                             if os.path.isdir(os.path.join(data_dir, d))])
        
        # Create class to index mapping
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        # Get all image paths and labels
        self.images = []
        self.labels = []
        
        for class_name in self.classes:
            class_dir = self.data_dir / class_name
            class_idx = self.class_to_idx[class_name]
            
            # Support multiple image formats
            for ext in ('*.jpg', '*.jpeg', '*.png'):
                for img_path in class_dir.glob(ext):
                    self.images.append(str(img_path))
                    self.labels.append(class_idx)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # Open image using PIL
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
            
        return image, label
    
    @staticmethod
    def get_default_transforms():
        """Default transforms for training"""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])

class ClassificationModel(nn.Module):
    """Modified model for 3-class classification"""
    
    def __init__(self, num_classes=3):
        super(ClassificationModel, self).__init__()
        # Load pretrained ResNet50
        self.model = models.resnet50(pretrained=True)
        
        # Modify the last fully connected layer for 3 classes
        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)

class Trainer:
    def __init__(self, model, train_loader, val_loader, device, 
                 learning_rate=0.001, num_epochs=50):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.num_epochs = num_epochs
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5)
        
        # Initialize metrics storage
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
        # Create results directory
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.results_dir = f'training_results_{self.timestamp}'
        os.makedirs(self.results_dir, exist_ok=True)
    
    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        return epoch_loss, epoch_acc
    
    def validate(self):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                
                running_loss += loss.item()
                
                # Calculate accuracy
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        val_loss = running_loss / len(self.val_loader)
        val_acc = 100. * correct / total
        return val_loss, val_acc
    
    def save_results(self):
        # Plot losses and accuracies
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(self.train_accuracies, label='Train Acc')
        plt.plot(self.val_accuracies, label='Val Acc')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'training_metrics.png'))
        plt.close()
        
        # Save metrics as JSON
        metrics = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies
        }
        
        with open(os.path.join(self.results_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f)
    
    def train(self):
        best_val_acc = 0.0
        
        for epoch in range(self.num_epochs):
            print(f'\nEpoch {epoch+1}/{self.num_epochs}')
            
            # Training
            train_loss, train_acc = self.train_epoch()
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            
            # Validation
            val_loss, val_acc = self.validate()
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            
            # Print metrics
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), 
                         os.path.join(self.results_dir, 'best_model.pth'))
            
            # Save intermediate results
            if (epoch + 1) % 5 == 0:
                self.save_results()
        
        # Save final results
        self.save_results()

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set your data directory path - MODIFY THIS PATH TO YOUR DATASET LOCATION
    data_dir = '/home/idrone2/Desktop/Kaggle_datasets/alzheimer'
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader = create_data_loaders(
        data_dir,
        batch_size=32,
        train_split=0.8
    )
    
    # Initialize model
    print("Initializing model...")
    model = ClassificationModel(num_classes=3).to(device)
    
    # Initialize trainer
    print("Setting up trainer...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=0.001,
        num_epochs=50
    )
    
    # Start training
    print("Starting training...")
    trainer.train()

if __name__ == '__main__':
    main()

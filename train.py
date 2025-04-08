import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

# Clear CUDA cache
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
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        d4 = self.dec4(torch.cat([nn.functional.interpolate(e4, scale_factor=2, mode='bilinear', align_corners=True), e3], dim=1))
        d3 = self.dec3(torch.cat([nn.functional.interpolate(d4, scale_factor=2, mode='bilinear', align_corners=True), e2], dim=1))
        d2 = self.dec2(torch.cat([nn.functional.interpolate(d3, scale_factor=2, mode='bilinear', align_corners=True), e1], dim=1))
        
        return self.dec1(d2)

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device='cuda'):
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print('-' * 20)

        train_loader_tqdm = tqdm(train_loader, desc="Training", leave=False)

        for inputs, _ in train_loader_tqdm:
            inputs = inputs.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, inputs)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            train_loader_tqdm.set_postfix({"Batch Loss": loss.item()})

        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        print(f"Train Loss: {train_loss:.4f}")

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_loader_tqdm = tqdm(val_loader, desc="Validation", leave=False)

        with torch.no_grad():
            for inputs, _ in val_loader_tqdm:
                inputs = inputs.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, inputs)
                val_loss += loss.item() * inputs.size(0)
                val_loader_tqdm.set_postfix({"Batch Loss": loss.item()})

        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)

        print(f"Validation Loss: {val_loss:.4f}")

    return train_losses, val_losses

def plot_losses(train_losses, val_losses, save_path):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.savefig(save_path)
    plt.close()

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Data loading and preprocessing
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    dataset_path = '/home/idrone2/Tea_pest/Tea-TJ'
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    dataset = ImageFolder(root=dataset_path, transform=transform)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # Model, loss, and optimizer
    model = UNet(in_channels=3, out_channels=3).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training
    train_losses, val_losses = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device=device)

    # Plot and save losses
    plot_losses(train_losses, val_losses, 'loss_plot.png')

    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, 'model_checkpoint.pth')

if __name__ == "__main__":
    main()
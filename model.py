import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        self.encoder1 = self.conv_block(in_channels, 64)   # 64
        self.encoder2 = self.conv_block(64, 128)           # 128
        self.encoder3 = self.conv_block(128, 256)          # 256
        self.encoder4 = self.conv_block(256, 512)          # 512
        self.bottleneck = self.conv_block(512, 1024)       # 1024

        self.decoder4 = self.upconv_block(1024, 512)       # 512
        self.decoder3 = self.upconv_block(512, 256)        # 256
        self.decoder2 = self.upconv_block(256, 128)        # 128
        self.decoder1 = self.upconv_block(128, out_channels)  # out_channels

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def upconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder path
        e1 = self.encoder1(x)  # (batch_size, 64, h/1, w/1)
        e2 = self.encoder2(e1)  # (batch_size, 128, h/2, w/2)
        e3 = self.encoder3(e2)  # (batch_size, 256, h/4, w/4)
        e4 = self.encoder4(e3)  # (batch_size, 512, h/8, w/8)

        # Bottleneck
        b = self.bottleneck(e4)  # (batch_size, 1024, h/16, w/16)

        # Decoder path
        d4 = self.decoder4(b)  # (batch_size, 512, h/8, w/8)
        d4 = torch.cat((d4, e4), dim=1)  # Concatenate (batch_size, 1024, h/8, w/8)

        d3 = self.decoder3(d4)  # (batch_size, 256, h/4, w/4)
        d3 = torch.cat((d3, e3), dim=1)  # Concatenate (batch_size, 512, h/4, w/4)

        d2 = self.decoder2(d3)  # (batch_size, 128, h/2, w/2)
        d2 = torch.cat((d2, e2), dim=1)  # Concatenate (batch_size, 256, h/2, w/2)

        d1 = self.decoder1(d2)  # (batch_size, out_channels, h, w)

        return d1

# Test the model
if __name__ == "__main__":
    model = UNet(in_channels=3, out_channels=3)
    x = torch.randn(1, 3, 256, 256)  # Example input
    output = model(x)
    print("Output shape:", output.shape)

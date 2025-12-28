"""Lightweight CNN model for audio classification."""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Convolutional block with Conv2d, BatchNorm, ReLU, and MaxPool."""

    def __init__(self, in_channels, out_channels, kernel_size=3, pool_size=2):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(pool_size)
        self.dropout = nn.Dropout2d(0.2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.dropout(x)
        return x


class LightweightCNN(nn.Module):
    """
    Lightweight CNN model optimized for RTX 3050 4GB. 

    Architecture:
    - 5 convolutional blocks with increasing channels
    - Global average pooling
    - Fully connected layers with dropout
    - Memory efficient design
    """

    def __init__(self, num_classes=4, dropout=0.3):
        super().__init__()

        # Convolutional blocks
        self.conv1 = ConvBlock(1, 32, kernel_size=3, pool_size=2)
        self.conv2 = ConvBlock(32, 64, kernel_size=3, pool_size=2)
        self.conv3 = ConvBlock(64, 128, kernel_size=3, pool_size=2)
        self.conv4 = ConvBlock(128, 256, kernel_size=3, pool_size=2)
        self.conv5 = ConvBlock(256, 256, kernel_size=3, pool_size=2)

        # Global average pooling
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Fully connected layers
        self.fc1 = nn.Linear(256, 128)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, num_classes)

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights using He initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn. init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m. weight, 0, 0.01)
                nn.init. constant_(m.bias, 0)

    def forward(self, x):
        """
        Forward pass. 

        Args:
            x: Input tensor of shape (batch_size, 1, n_mels, time_steps)

        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        # Convolutional blocks
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        # Global average pooling
        x = self.gap(x)
        x = torch.flatten(x, 1)

        # Fully connected layers
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x


def count_parameters(model):
    """Count the number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__": 
    # Test the model
    model = LightweightCNN(num_classes=4)
    x = torch.randn(8, 1, 128, 313)  # Batch of 8, 1 channel, 128 mel bins, 313 time steps
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape:  {output.shape}")
    print(f"Total parameters: {count_parameters(model):,}")

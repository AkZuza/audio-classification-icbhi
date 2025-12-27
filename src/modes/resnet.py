"""Compact ResNet model for audio classification."""

import torch
import torch.nn as nn
import torchvision.models as models


class CompactResNet(nn.Module):
    """
    Compact ResNet18 model optimized for audio spectrograms.

    Architecture:
    - Modified ResNet18 with reduced channels
    - Custom first layer for single-channel input
    - Optional pretrained weights
    - Memory efficient for RTX 3050 4GB
    """

    def __init__(self, num_classes=4, pretrained=False, dropout=0.3):
        super().__init__()

        # Load ResNet18
        if pretrained:
            self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        else:
            self.resnet = models. resnet18(weights=None)

        # Modify first conv layer for single-channel input (mel-spectrogram)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Replace the final fully connected layer
        num_features = self.resnet.fc. in_features
        self.resnet.fc = nn.Sequential(
            nn. Dropout(dropout),
            nn.Linear(num_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(256, num_classes),
        )

        self._initialize_first_layer()

    def _initialize_first_layer(self):
        """Initialize the modified first convolutional layer."""
        nn.init.kaiming_normal_(self.resnet.conv1.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        """
        Forward pass. 

        Args:
            x: Input tensor of shape (batch_size, 1, n_mels, time_steps)

        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        return self.resnet(x)


def count_parameters(model):
    """Count the number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the model
    model = CompactResNet(num_classes=4, pretrained=False)
    x = torch.randn(8, 1, 128, 313)  # Batch of 8, 1 channel, 128 mel bins, 313 time steps
    output = model(x)
    print(f"Input shape: {x. shape}")
    print(f"Output shape: {output.shape}")
    print(f"Total parameters: {count_parameters(model):,}")

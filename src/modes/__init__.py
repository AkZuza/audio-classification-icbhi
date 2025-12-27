"""Model architectures for audio classification."""

from . cnn import LightweightCNN
from .resnet import CompactResNet

__all__ = ["LightweightCNN", "CompactResNet"]

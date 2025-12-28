"""Configuration utilities."""

import yaml
import torch
import numpy as np
import random


def load_config(config_path="config.yaml"):
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to configuration file

    Returns:
        Configuration dictionary
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def set_seed(seed=42):
    """
    Set random seed for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends. cudnn.benchmark = False


def get_device(use_cuda=True):
    """
    Get device for training/inference.

    Args:
        use_cuda: Whether to use CUDA if available

    Returns:
        torch.device
    """
    if use_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


def save_checkpoint(model, optimizer, epoch, loss, path):
    """
    Save model checkpoint.

    Args:
        model: PyTorch model
        optimizer:  Optimizer
        epoch: Current epoch
        loss: Current loss
        path: Save path
    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }
    torch.save(checkpoint, path)


def load_checkpoint(model, optimizer, path, device):
    """
    Load model checkpoint.

    Args:
        model: PyTorch model
        optimizer: Optimizer
        path:  Checkpoint path
        device: Device to load checkpoint

    Returns:
        Tuple of (model, optimizer, epoch, loss)
    """
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer: 
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint. get("epoch", 0)
    loss = checkpoint.get("loss", 0.0)
    return model, optimizer, epoch, loss

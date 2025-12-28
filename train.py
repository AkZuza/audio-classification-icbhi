"""Main training script for audio classification."""

import argparse
import yaml
from pathlib import Path
import torch

from src.data.dataset import ICBHIDataset
from src.models.cnn import LightweightCNN
from src.models.resnet import CompactResNet
from src.training.trainer import Trainer
from src.utils.config import load_config, set_seed, get_device
from src.utils.metrics import plot_training_history


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train audio classification model")
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to configuration file"
    )
    parser.add_argument("--model", type=str, choices=["cnn", "resnet"], help="Model architecture")
    parser.add_argument("--epochs", type=int, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, help="Batch size")
    parser.add_argument("--learning-rate", type=float, help="Learning rate")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], help="Device to use")
    return parser.parse_args()


def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()

    # Load configuration
    config = load_config(args.config)

    # Override config with command-line arguments
    if args.model: 
        config["model"]["architecture"] = args.model
    if args.epochs:
        config["training"]["epochs"] = args.epochs
    if args.batch_size:
        config["training"]["batch_size"] = args.batch_size
    if args.learning_rate:
        config["training"]["learning_rate"] = args.learning_rate
    if args.device:
        config["device"]["use_cuda"] = args.device == "cuda"

    # Set random seed
    set_seed(config. get("seed", 42))

    # Get device
    device = get_device(config["device"]["use_cuda"])

    # Print configuration
    print("\n" + "=" * 60)
    print("TRAINING CONFIGURATION")
    print("=" * 60)
    print(f"Model: {config['model']['architecture']}")
    print(f"Epochs: {config['training']['epochs']}")
    print(f"Batch size: {config['training']['batch_size']}")
    print(f"Learning rate: {config['training']['learning_rate']}")
    print(f"Mixed precision: {config['training']['mixed_precision']}")
    print(f"Gradient accumulation:  {config['training']['gradient_accumulation_steps']}")
    print(f"Dataset: {config['data']['dataset_path']}")
    print(f"Device: {device}")
    print("=" * 60 + "\n")

    # Create datasets
    print("Loading datasets...")
    train_dataset = ICBHIDataset(
        root_dir=config["data"]["dataset_path"],
        split="train",
        config=config,
        augment=config["data"]["augmentation"],
    )

    val_dataset = ICBHIDataset(
        root_dir=config["data"]["dataset_path"], split="val", config=config, augment=False
    )

    # Create model
    print(f"\nCreating {config['model']['architecture']. upper()} model...")
    if config["model"]["architecture"] == "cnn":
        model = LightweightCNN(
            num_classes=config["model"]["num_classes"], dropout=config["model"]["dropout"]
        )
    else:  # resnet
        model = CompactResNet(
            num_classes=config["model"]["num_classes"],
            pretrained=False,
            dropout=config["model"]["dropout"],
        )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p. numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Create trainer
    trainer = Trainer(
        model=model, train_dataset=train_dataset, val_dataset=val_dataset, config=config, device=device
    )

    # Train
    history = trainer.train()

    # Plot training history
    print("\nPlotting training history...")
    plot_path = Path(config["training"]["checkpoint_dir"]) / "training_history.png"
    plot_training_history(history, save_path=plot_path)

    print(f"\n✓ Training completed successfully!")
    print(f"✓ Best model saved to: {config['training']['checkpoint_dir']}/best_model.pt")
    print(f"✓ Training history saved to: {plot_path}")


if __name__ == "__main__":
    main()

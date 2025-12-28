"""Validation script for audio classification."""

import argparse
import torch
from pathlib import Path

from src.data.dataset import ICBHIDataset
from src.models.cnn import LightweightCNN
from src. models.resnet import CompactResNet
from src.training.validation import Validator
from src.utils.config import load_config, get_device
from src.utils. metrics import (
    calculate_metrics,
    print_metrics,
    plot_confusion_matrix,
    plot_roc_curves,
)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Validate audio classification model")
    parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to configuration file"
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "val", "test"],
        default="test",
        help="Dataset split to validate",
    )
    parser.add_argument(
        "--device", type=str, choices=["cuda", "cpu"], help="Device to use"
    )
    return parser.parse_args()


def main():
    """Main validation function."""
    # Parse arguments
    args = parse_args()

    # Load configuration
    try:
        # Try to load config from checkpoint
        checkpoint = torch.load(args.model, map_location="cpu")
        config = checkpoint. get("config")
        if config is None: 
            # Fall back to config file
            config = load_config(args.config)
    except:
        config = load_config(args.config)

    # Override device if specified
    if args.device:
        config["device"]["use_cuda"] = args.device == "cuda"

    # Get device
    device = get_device(config["device"]["use_cuda"])

    print("\n" + "=" * 60)
    print("VALIDATION CONFIGURATION")
    print("=" * 60)
    print(f"Model checkpoint: {args.model}")
    print(f"Dataset split: {args.split}")
    print(f"Device: {device}")
    print("=" * 60 + "\n")

    # Load dataset
    print(f"Loading {args.split} dataset...")
    dataset = ICBHIDataset(
        root_dir=config["data"]["dataset_path"], split=args.split, config=config, augment=False
    )

    # Create model
    print(f"\nCreating {config['model']['architecture'].upper()} model...")
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

    # Load checkpoint
    print(f"Loading model from {args.model}...")
    checkpoint = torch.load(args.model, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"✓ Loaded checkpoint from epoch {checkpoint. get('epoch', 'unknown')}")

    # Create validator
    validator = Validator(model=model, dataset=dataset, config=config, device=device)

    # Run validation
    print("\nRunning validation...")
    y_true, y_pred, y_prob = validator.validate()

    # Calculate metrics
    class_names = config["classes"]
    metrics = calculate_metrics(y_true, y_pred, y_prob, class_names)

    # Print metrics
    print_metrics(metrics, class_names)

    # Plot confusion matrix
    output_dir = Path("validation_results")
    output_dir.mkdir(exist_ok=True)

    cm_path = output_dir / f"confusion_matrix_{args.split}.png"
    plot_confusion_matrix(y_true, y_pred, class_names, save_path=cm_path)

    # Plot ROC curves
    roc_path = output_dir / f"roc_curves_{args.split}.png"
    plot_roc_curves(y_true, y_prob, class_names, save_path=roc_path)

    print(f"\n✓ Validation completed successfully!")
    print(f"✓ Results saved to: {output_dir}")


if __name__ == "__main__":
    main()

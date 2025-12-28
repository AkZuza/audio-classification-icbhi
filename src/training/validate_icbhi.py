"""Validation script with ICBHI 2017 Challenge scoring."""

import argparse
import torch
from pathlib import Path
import numpy as np

from src.data.dataset_segmented import ICBHISegmentedDataset
from src.models.cnn import LightweightCNN
from src.models.resnet import CompactResNet
from src.training.validation import Validator
from src.utils.config import load_config, get_device
from src.utils.icbhi_metrics import (
    calculate_icbhi_score,
    print_icbhi_metrics,
    plot_icbhi_metrics,
    calculate_detailed_confusion_metrics,
    plot_detailed_confusion_matrix
)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Validate with ICBHI scoring")
    parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument(
        "--config",
        type=str,
        default="config_segmented.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "val", "test"],
        default="test",
        help="Dataset split to validate"
    )
    parser.add_argument(
        "--device", type=str, choices=["cuda", "cpu"], help="Device to use"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="validation_results",
        help="Directory to save results"
    )
    return parser.parse_args()


def main():
    """Main validation function."""
    args = parse_args()

    # Load configuration
    try:
        checkpoint = torch.load(args.model, map_location="cpu")
        config = checkpoint. get("config")
        if config is None:
            config = load_config(args.config)
    except: 
        config = load_config(args.config)

    # Override device if specified
    if args.device:
        config["device"]["use_cuda"] = args.device == "cuda"

    # Get device
    device = get_device(config["device"]["use_cuda"])

    print("\n" + "=" * 70)
    print("ICBHI 2017 CHALLENGE VALIDATION")
    print("=" * 70)
    print(f"Model checkpoint: {args.model}")
    print(f"Dataset split: {args.split}")
    print(f"Device: {device}")
    print("=" * 70 + "\n")

    # Load dataset
    print(f"Loading {args.split} dataset...")
    dataset = ICBHISegmentedDataset(
        root_dir=config["data"]["dataset_path"],
        split=args.split,
        config=config,
        augment=False
    )

    # Create model
    print(f"\nCreating {config['model']['architecture']. upper()} model...")
    if config["model"]["architecture"] == "cnn":
        model = LightweightCNN(
            num_classes=config["model"]["num_classes"],
            dropout=config["model"]["dropout"]
        )
    else:
        model = CompactResNet(
            num_classes=config["model"]["num_classes"],
            pretrained=False,
            dropout=config["model"]["dropout"]
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

    # Get class names
    class_names = config["classes"]

    # Calculate ICBHI score
    print("\nCalculating ICBHI metrics...")
    icbhi_metrics = calculate_icbhi_score(y_true, y_pred, class_names)

    # Print ICBHI metrics
    print_icbhi_metrics(icbhi_metrics, class_names)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Plot ICBHI metrics
    icbhi_plot_path = output_dir / f"icbhi_metrics_{args.split}.png"
    plot_icbhi_metrics(icbhi_metrics, class_names, save_path=icbhi_plot_path)

    # Calculate detailed confusion metrics
    detailed_metrics, cm = calculate_detailed_confusion_metrics(y_true, y_pred, class_names)

    # Plot confusion matrix
    cm_path = output_dir / f"confusion_matrix_{args.split}.png"
    plot_detailed_confusion_matrix(cm, class_names, save_path=cm_path)

    # Print detailed metrics
    print("\n" + "=" * 70)
    print("DETAILED CONFUSION MATRIX METRICS")
    print("=" * 70)
    print(f"{'Class':<15} {'Precision':<12} {'Recall(Sens)':<15} {'F1-Score': <12}")
    print("-" * 70)

    for class_name in class_names:
        metrics = detailed_metrics[class_name]
        print(
            f"{class_name:<15} "
            f"{metrics['precision']:<12.4f} "
            f"{metrics['sensitivity']:<15.4f} "
            f"{metrics['f1_score']:<12.4f}"
        )

    print("=" * 70)

    # Save results to file
    results_file = output_dir / f"icbhi_results_{args.split}.txt"
    with open(results_file, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("ICBHI 2017 CHALLENGE RESULTS\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Model: {args. model}\n")
        f.write(f"Split: {args.split}\n")
        f.write(f"Total samples: {len(y_true)}\n\n")
        
        f.write(f"ICBHI Score: {icbhi_metrics['icbhi_score']:.4f}\n")
        f.write(f"Average Sensitivity: {icbhi_metrics['avg_sensitivity']:.4f}\n")
        f.write(f"Average Specificity: {icbhi_metrics['avg_specificity']:.4f}\n")
        f.write(f"Overall Accuracy: {icbhi_metrics['accuracy']:.4f}\n\n")
        
        f.write("Per-Class Metrics:\n")
        f.write("-" * 70 + "\n")
        for class_name in class_names: 
            class_metrics = icbhi_metrics['per_class_metrics'][class_name]
            f.write(f"\n{class_name.upper()}:\n")
            f.write(f"  Sensitivity: {class_metrics['sensitivity']:.4f}\n")
            f.write(f"  Specificity: {class_metrics['specificity']:.4f}\n")
            f.write(f"  Harmonic Score: {class_metrics['harmonic_score']:.4f}\n")

    print(f"\n✓ Validation completed successfully!")
    print(f"✓ Results saved to: {output_dir}")
    print(f"✓ ICBHI Score: {icbhi_metrics['icbhi_score']:.4f}")


if __name__ == "__main__":
    main()

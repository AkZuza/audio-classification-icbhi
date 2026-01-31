"""
Generate confusion matrix from TensorBoard event logs. 
Reads the latest training run and creates a confusion matrix visualization.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from datetime import datetime
import torch

# Try to import tensorboard
try:
    from tensorboard.backend.event_processing import event_accumulator
except ImportError:
    print("⚠️  TensorBoard not installed. Install with: pip install tensorboard")
    exit(1)


def find_latest_run(log_dir='runs'):
    """Find the latest TensorBoard run directory."""
    log_path = Path(log_dir)
    
    if not log_path.exists():
        raise ValueError(f"Log directory not found: {log_dir}")
    
    # Find all subdirectories (each run)
    runs = [d for d in log_path. iterdir() if d.is_dir()]
    
    if not runs:
        raise ValueError(f"No runs found in {log_dir}")
    
    # Sort by modification time
    latest_run = max(runs, key=lambda x:  x.stat().st_mtime)
    
    print(f"Found latest run: {latest_run}")
    return latest_run


def load_event_file(run_dir):
    """Load TensorBoard event file from run directory."""
    event_files = list(Path(run_dir).glob('events. out.tfevents.*'))
    
    if not event_files:
        raise ValueError(f"No event files found in {run_dir}")
    
    # Use the latest event file
    event_file = max(event_files, key=lambda x: x.stat().st_mtime)
    
    print(f"Loading event file: {event_file. name}")
    
    # Load events
    ea = event_accumulator.EventAccumulator(str(event_file))
    ea.Reload()
    
    return ea


def generate_confusion_matrix_from_checkpoint(checkpoint_path, val_dataset, device='cuda'):
    """
    Generate confusion matrix by running validation on a checkpoint.
    
    Args:
        checkpoint_path:  Path to model checkpoint
        val_dataset:  Validation dataset
        device: Device to use
    
    Returns:
        Confusion matrix as numpy array
    """
    from src.models.cnn import LightweightCNN
    from src. models.resnet import CompactResNet
    from torch.utils.data import DataLoader
    from sklearn.metrics import confusion_matrix
    
    print(f"\nLoading checkpoint: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    config = checkpoint['config']
    
    # Create model
    if config['model']['architecture'] == 'cnn':
        model = LightweightCNN(
            num_classes=config['model']['num_classes'],
            dropout=config['model']['dropout']
        )
    else:
        model = CompactResNet(
            num_classes=config['model']['num_classes'],
            pretrained=False,
            dropout=config['model']['dropout']
        )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Create data loader
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4
    )
    
    # Run validation
    print("Running validation to generate confusion matrix...")
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in val_loader: 
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels. numpy())
    
    # Generate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    return cm, config['classes']


def plot_confusion_matrix(cm, class_names, save_path=None, show=True):
    """
    Plot confusion matrix with percentages.
    
    Args:
        cm: Confusion matrix
        class_names: List of class names
        save_path: Path to save figure
        show: Whether to display the plot
    """
    # Calculate percentages
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot heatmap
    sns.heatmap(
        cm,
        annot=False,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count'},
        ax=ax,
        square=True
    )
    
    # Add custom annotations (count + percentage)
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            count = cm[i, j]
            percentage = cm_percent[i, j]
            
            # Choose text color based on background
            text_color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
            
            # Add text
            text = f'{count}\n({percentage:.1f}%)'
            ax.text(
                j + 0.5, i + 0.5, text,
                ha='center', va='center',
                color=text_color,
                fontsize=13,
                fontweight='bold'
            )
    
    # Formatting
    ax.set_title('Confusion Matrix', fontsize=18, fontweight='bold', pad=20)
    ax.set_ylabel('True Label', fontsize=14, fontweight='bold')
    ax.set_xlabel('Predicted Label', fontsize=14, fontweight='bold')
    
    # Rotate labels
    plt.setp(ax. get_xticklabels(), rotation=45, ha='right', fontsize=12)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"\n✓ Confusion matrix saved to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_normalized_confusion_matrix(cm, class_names, save_path=None, show=True):
    """
    Plot normalized confusion matrix (percentages only).
    
    Args:
        cm: Confusion matrix
        class_names: List of class names
        save_path: Path to save figure
        show: Whether to display the plot
    """
    # Normalize
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot heatmap
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt='.2%',
        cmap='RdYlGn',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Percentage', 'format': '%.0f%%'},
        ax=ax,
        square=True,
        vmin=0,
        vmax=1,
        linewidths=1,
        linecolor='gray'
    )
    
    # Formatting
    ax.set_title('Normalized Confusion Matrix (Percentages)', 
                fontsize=18, fontweight='bold', pad=20)
    ax.set_ylabel('True Label', fontsize=14, fontweight='bold')
    ax.set_xlabel('Predicted Label', fontsize=14, fontweight='bold')
    
    # Rotate labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=12)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Normalized confusion matrix saved to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def print_classification_report(cm, class_names):
    """Print detailed classification metrics."""
    print("\n" + "=" * 70)
    print("CLASSIFICATION REPORT")
    print("=" * 70)
    
    total_samples = cm.sum()
    
    print(f"\nTotal samples: {int(total_samples)}")
    print(f"\n{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print("-" * 70)
    
    for i, class_name in enumerate(class_names):
        TP = cm[i, i]
        FP = cm[: , i].sum() - TP
        FN = cm[i, :].sum() - TP
        
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        support = cm[i, : ].sum()
        
        print(f"{class_name:<15} {precision:<12.4f} {recall:<12.4f} {f1:<12.4f} {int(support):<10}")
    
    # Overall accuracy
    accuracy = np.trace(cm) / cm.sum()
    print("-" * 70)
    print(f"{'Overall Accuracy':<15} {accuracy:. 4f}")
    print("=" * 70)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Generate confusion matrix from TensorBoard logs"
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="runs",
        help="TensorBoard log directory (default: runs)"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best_model.pt",
        help="Path to model checkpoint (default: checkpoints/best_model.pt)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config_segmented.yaml",
        help="Path to config file (default: config_segmented.yaml)"
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=['train', 'val', 'test'],
        default='test',
        help="Dataset split to use (default: test)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="confusion_matrices",
        help="Output directory (default: confusion_matrices)"
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=['cuda', 'cpu'],
        default='cuda',
        help="Device to use"
    )
    parser.add_argument(
        "--no-display",
        action='store_true',
        help="Don't display plots"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 70)
    print("CONFUSION MATRIX GENERATOR")
    print("=" * 70)
    
    # Check if checkpoint exists
    if not Path(args.checkpoint).exists():
        print(f"\n❌ Checkpoint not found: {args.checkpoint}")
        print("Please specify a valid checkpoint with --checkpoint")
        return
    
    # Load config
    import yaml
    with open(args. config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load dataset
    print(f"\nLoading {args.split} dataset...")
    
    try:
        from src.data.dataset_segmented import ICBHISegmentedDataset
        
        dataset = ICBHISegmentedDataset(
            root_dir=config['data']['dataset_path'],
            split=args.split,
            config=config,
            augment=False
        )
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        print("\nMake sure you have:")
        print("1. Preprocessed the ICBHI dataset (python preprocess_icbhi.py)")
        print("2. Updated the config file with correct dataset path")
        return
    
    # Generate confusion matrix
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    try:
        cm, class_names = generate_confusion_matrix_from_checkpoint(
            args.checkpoint,
            dataset,
            device=device
        )
    except Exception as e: 
        print(f"❌ Failed to generate confusion matrix: {e}")
        return
    
    # Print classification report
    print_classification_report(cm, class_names)
    
    # Plot confusion matrices
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Regular confusion matrix
    cm_path = output_dir / f"confusion_matrix_{args.split}_{timestamp}.png"
    plot_confusion_matrix(
        cm,
        class_names,
        save_path=str(cm_path),
        show=not args.no_display
    )
    
    # Normalized confusion matrix
    cm_norm_path = output_dir / f"confusion_matrix_normalized_{args.split}_{timestamp}. png"
    plot_normalized_confusion_matrix(
        cm,
        class_names,
        save_path=str(cm_norm_path),
        show=not args.no_display
    )
    
    # Save confusion matrix as numpy array
    cm_npy_path = output_dir / f"confusion_matrix_{args.split}_{timestamp}.npy"
    np.save(cm_npy_path, cm)
    print(f"✓ Confusion matrix array saved to: {cm_npy_path}")
    
    # Save as CSV
    import pandas as pd
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_csv_path = output_dir / f"confusion_matrix_{args.split}_{timestamp}.csv"
    cm_df.to_csv(cm_csv_path)
    print(f"✓ Confusion matrix CSV saved to: {cm_csv_path}")
    
    print(f"\n✓ All results saved to: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()

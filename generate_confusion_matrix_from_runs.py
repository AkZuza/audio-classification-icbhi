"""
Generate confusion matrix from the latest TensorBoard run in runs/ folder.
Works with your specific directory structure.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from datetime import datetime
import torch
from sklearn.metrics import confusion_matrix


def find_latest_event_file(runs_dir='runs'):
    """Find the most recent event file in runs directory."""
    runs_path = Path(runs_dir)
    
    if not runs_path.exists():
        raise ValueError(f"Runs directory not found: {runs_dir}")
    
    # Find all event files
    event_files = list(runs_path.glob('events.out.tfevents.*'))
    
    if not event_files:
        raise ValueError(f"No event files found in {runs_dir}")
    
    # Sort by modification time and get the latest
    latest_event = max(event_files, key=lambda x: x.stat().st_mtime)
    
    print(f"✓ Found latest event file: {latest_event. name}")
    print(f"  Modified: {datetime.fromtimestamp(latest_event.stat().st_mtime)}")
    
    return latest_event


def generate_confusion_matrix_from_checkpoint(checkpoint_path, val_dataset, device='cuda'):
    """
    Generate confusion matrix by running validation. 
    
    Args:
        checkpoint_path: Path to model checkpoint
        val_dataset:  Validation dataset
        device: Device to use
    
    Returns:
        Confusion matrix and class names
    """
    from src.models.cnn import LightweightCNN
    from src. models.resnet import CompactResNet
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    
    print(f"\n📦 Loading checkpoint: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    config = checkpoint. get('config')
    
    if config is None:
        raise ValueError("Config not found in checkpoint.  Please retrain your model.")
    
    # Create model
    print(f"   Architecture: {config['model']['architecture']}")
    
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
    
    print(f"   Classes: {config['classes']}")
    print(f"   Device: {device}")
    
    # Create data loader
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Run validation
    print(f"\n🔍 Running validation on {len(val_dataset)} samples...")
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Validating"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels. numpy())
    
    # Generate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    return cm, config['classes']


def plot_confusion_matrix(cm, class_names, save_path=None, show=True):
    """Plot confusion matrix with counts and percentages."""
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
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
        square=True,
        linewidths=1,
        linecolor='gray'
    )
    
    # Add custom annotations
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            count = cm[i, j]
            percentage = cm_percent[i, j]
            
            text_color = 'white' if cm[i, j] > cm. max() / 2 else 'black'
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
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=12)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Confusion matrix saved to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_normalized_confusion_matrix(cm, class_names, save_path=None, show=True):
    """Plot normalized confusion matrix."""
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt='.1%',
        cmap='RdYlGn',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Percentage'},
        ax=ax,
        square=True,
        vmin=0,
        vmax=1,
        linewidths=1,
        linecolor='gray'
    )
    
    ax.set_title('Normalized Confusion Matrix', fontsize=18, fontweight='bold', pad=20)
    ax.set_ylabel('True Label', fontsize=14, fontweight='bold')
    ax.set_xlabel('Predicted Label', fontsize=14, fontweight='bold')
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=12)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Normalized confusion matrix saved to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt. close()


def print_classification_report(cm, class_names):
    """Print detailed metrics."""
    print("\n" + "=" * 70)
    print("CLASSIFICATION REPORT")
    print("=" * 70)
    
    total = cm.sum()
    print(f"\nTotal samples: {int(total)}")
    
    print(f"\n{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print("-" * 70)
    
    precisions = []
    recalls = []
    f1s = []
    
    for i, class_name in enumerate(class_names):
        TP = cm[i, i]
        FP = cm[: , i].sum() - TP
        FN = cm[i, :].sum() - TP
        
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        support = cm[i, : ].sum()
        
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        
        print(f"{class_name:<15} {precision: <12.4f} {recall:<12.4f} {f1:<12.4f} {int(support):<10}")
    
    # Weighted averages
    supports = cm.sum(axis=1)
    weighted_precision = np.average(precisions, weights=supports)
    weighted_recall = np.average(recalls, weights=supports)
    weighted_f1 = np.average(f1s, weights=supports)
    
    print("-" * 70)
    print(f"{'Accuracy':<15} {np.trace(cm)/cm.sum():.4f}")
    print(f"{'Weighted Avg':<15} {weighted_precision: <12.4f} {weighted_recall:<12.4f} {weighted_f1:<12.4f}")
    print("=" * 70)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Generate confusion matrix from TensorBoard runs"
    )
    parser.add_argument(
        "--runs-dir",
        type=str,
        default="runs",
        help="TensorBoard runs directory (default: runs)"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best_model.pt",
        help="Path to checkpoint (default: checkpoints/best_model.pt)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config_segmented.yaml",
        help="Config file (default: config_segmented.yaml)"
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=['train', 'val', 'test'],
        default='test',
        help="Dataset split (default: test)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="confusion_matrices",
        help="Output directory"
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=['cuda', 'cpu'],
        default='cuda'
    )
    parser.add_argument(
        "--no-display",
        action='store_true'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("CONFUSION MATRIX GENERATOR")
    print("=" * 70)
    
    # Find latest run
    try:
        latest_event = find_latest_event_file(args.runs_dir)
    except Exception as e:
        print(f"\n❌ Error:  {e}")
        return
    
    # Check checkpoint
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"\n❌ Checkpoint not found: {checkpoint_path}")
        print("\nAvailable checkpoints:")
        checkpoint_dir = Path("checkpoints")
        if checkpoint_dir.exists():
            for cp in checkpoint_dir.glob("*.pt"):
                print(f"  - {cp}")
        return
    
    # Load config
    import yaml
    
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"\n❌ Config not found: {config_path}")
        return
    
    with open(config_path, 'r') as f:
        config = yaml. safe_load(f)
    
    # Load dataset
    print(f"\n📂 Loading {args.split} dataset...")
    print(f"   Dataset path: {config['data']['dataset_path']}")
    
    try:
        from src.data.dataset_segmented import ICBHISegmentedDataset
        
        dataset = ICBHISegmentedDataset(
            root_dir=config['data']['dataset_path'],
            split=args.split,
            config=config,
            augment=False
        )
    except Exception as e:
        print(f"\n❌ Failed to load dataset: {e}")
        print("\nMake sure:")
        print("1. Dataset is preprocessed:  python preprocess_icbhi. py")
        print("2. Config has correct dataset path")
        return
    
    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # Generate confusion matrix
    try:
        cm, class_names = generate_confusion_matrix_from_checkpoint(
            checkpoint_path,
            dataset,
            device=device
        )
    except Exception as e:
        print(f"\n❌ Failed:  {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Print report
    print_classification_report(cm, class_names)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Plot confusion matrices
    cm_path = output_dir / f"confusion_matrix_{args.split}_{timestamp}.png"
    plot_confusion_matrix(cm, class_names, save_path=str(cm_path), show=not args.no_display)
    
    cm_norm_path = output_dir / f"confusion_matrix_normalized_{args.split}_{timestamp}.png"
    plot_normalized_confusion_matrix(cm, class_names, save_path=str(cm_norm_path), show=not args.no_display)
    
    # Save as numpy
    np.save(output_dir / f"confusion_matrix_{args.split}_{timestamp}. npy", cm)
    
    # Save as CSV
    import pandas as pd
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_df.to_csv(output_dir / f"confusion_matrix_{args.split}_{timestamp}. csv")
    
    print(f"\n✓ All results saved to: {output_dir}")
    print("=" * 70)


if __name__ == "__main__": 
    main()

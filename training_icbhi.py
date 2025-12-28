"""Training script with ICBHI score tracking."""

import argparse
import yaml
from pathlib import Path
import torch

from src.data.dataset_segmented import ICBHISegmentedDataset
from src.models.cnn import LightweightCNN
from src.models.resnet import CompactResNet
from src.training.trainer_icbhi import TrainerWithICBHI
from src. utils.config import load_config, set_seed, get_device
from src.utils.metrics import plot_training_history
import matplotlib.pyplot as plt


def plot_icbhi_history(history, save_path=None):
    """Plot ICBHI score history."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss
    axes[0, 0]. plot(history['train_loss'], label='Train Loss')
    axes[0, 0].plot(history['val_loss'], label='Val Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # Accuracy
    axes[0, 1].plot(history['train_acc'], label='Train Acc')
    axes[0, 1].plot(history['val_acc'], label='Val Acc')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].set_title('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    # ICBHI Score
    axes[1, 0].plot(history['icbhi_score'], label='ICBHI Score', linewidth=2, color='green')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('ICBHI Score')
    axes[1, 0].set_title('ICBHI Score (Main Metric)')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    
    # Sensitivity & Specificity
    axes[1, 1].plot(history['sensitivity'], label='Sensitivity')
    axes[1, 1].plot(history['specificity'], label='Specificity')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].set_title('Sensitivity and Specificity')
    axes[1, 1].legend()
    axes[1, 1]. grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train with ICBHI scoring")
    parser.add_argument(
        "--config",
        type=str,
        default="config_segmented.yaml",
        help="Path to configuration file"
    )
    parser.add_argument("--model", type=str, choices=["cnn", "resnet"], help="Model architecture")
    parser.add_argument("--epochs", type=int, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, help="Batch size")
    parser.add_argument("--learning-rate", type=float, help="Learning rate")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], help="Device to use")
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    config = load_config(args.config)

    # Override config
    if args.model: 
        config['model']['architecture'] = args.model
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.learning_rate:
        config['training']['learning_rate'] = args.learning_rate
    if args.device:
        config['device']['use_cuda'] = args.device == 'cuda'

    # Set seed
    set_seed(config. get('seed', 42))

    # Get device
    device = get_device(config['device']['use_cuda'])

    print("\n" + "=" * 70)
    print("TRAINING WITH ICBHI 2017 CHALLENGE SCORING")
    print("=" * 70)
    print(f"Model: {config['model']['architecture']}")
    print(f"Dataset: {config['data']['dataset_path']}")
    print(f"Evaluation Metric: ICBHI Score")
    print("=" * 70 + "\n")

    # Load datasets
    print("Loading datasets...")
    train_dataset = ICBHISegmentedDataset(
        root_dir=config['data']['dataset_path'],
        split='train',
        config=config,
        augment=config['data']['augmentation']
    )

    val_dataset = ICBHISegmentedDataset(
        root_dir=config['data']['dataset_path'],
        split='val',
        config=config,
        augment=False
    )

    # Create model
    print(f"\nCreating {config['model']['architecture']. upper()} model...")
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

    # Create trainer
    trainer = TrainerWithICBHI(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=config,
        device=device
    )

    # Train
    history = trainer.train()

    # Plot training history
    print("\nPlotting training history...")
    plot_path = Path(config['training']['checkpoint_dir']) / "training_history_icbhi.png"
    plot_icbhi_history(history, save_path=plot_path)

    print(f"\n✓ Training completed!")
    print(f"✓ Best ICBHI Score:  {trainer.best_icbhi_score:.4f}")
    print(f"✓ Best model saved to: {config['training']['checkpoint_dir']}/best_model.pt")


if __name__ == "__main__": 
    main()

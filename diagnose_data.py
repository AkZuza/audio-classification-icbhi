"""Diagnose data loading and preprocessing issues."""

import torch
import yaml
import matplotlib.pyplot as plt
import numpy as np
from src.data.dataset import ICBHIDataset
from torch.utils.data import DataLoader


def diagnose_dataset():
    """Check if data is loading correctly."""
    
    # Load config
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    print("="*60)
    print("DATA DIAGNOSTICS")
    print("="*60)
    
    # Load dataset
    print("\n1. Loading dataset...")
    try:
        dataset = ICBHIDataset(
            root_dir=config["data"]["dataset_path"],
            split="train",
            config=config,
            augment=False
        )
        print(f"✓ Dataset loaded:  {len(dataset)} samples")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return
    
    # Check class distribution
    print("\n2. Checking class distribution...")
    labels = [dataset.data[i][1] for i in range(len(dataset))]
    unique, counts = np.unique(labels, return_counts=True)
    
    class_names = config["classes"]
    print(f"\nClass distribution:")
    for cls, count in zip(unique, counts):
        print(f"  {class_names[cls]}: {count} ({100*count/len(labels):.1f}%)")
    
    # Check if classes are imbalanced
    if max(counts) / min(counts) > 5:
        print("\n⚠ WARNING:  Highly imbalanced dataset detected!")
        print("  Consider using weighted loss or class balancing")
    
    # Check a sample
    print("\n3. Checking sample data...")
    mel_spec, label = dataset[0]
    
    print(f"  Mel-spectrogram shape: {mel_spec.shape}")
    print(f"  Mel-spectrogram dtype: {mel_spec.dtype}")
    print(f"  Label: {label} ({class_names[label]})")
    print(f"  Mel-spec range: [{mel_spec.min():.2f}, {mel_spec.max():.2f}]")
    print(f"  Mel-spec mean: {mel_spec.mean():.2f}")
    print(f"  Mel-spec std: {mel_spec.std():.2f}")
    
    # Check for NaN or Inf
    if torch.isnan(mel_spec).any():
        print("  ❌ NaN values detected in mel-spectrogram!")
    if torch.isinf(mel_spec).any():
        print("  ❌ Inf values detected in mel-spectrogram!")
    
    # Visualize samples
    print("\n4. Visualizing samples...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    
    for idx in range(6):
        if idx < len(dataset):
            mel_spec, label = dataset[idx]
            ax = axes[idx]
            
            # Remove batch dimension if present
            if mel_spec. dim() == 3:
                mel_spec = mel_spec.squeeze(0)
            
            im = ax.imshow(mel_spec.numpy(), aspect='auto', origin='lower', cmap='viridis')
            ax.set_title(f"Sample {idx}:  {class_names[label]}")
            ax.set_xlabel("Time")
            ax.set_ylabel("Mel Frequency")
            plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    plt.savefig("data_samples.png", dpi=150, bbox_inches='tight')
    print("  ✓ Saved visualization to:  data_samples.png")
    
    # Check DataLoader
    print("\n5. Testing DataLoader...")
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    batch = next(iter(loader))
    inputs, labels = batch
    
    print(f"  Batch input shape: {inputs.shape}")
    print(f"  Batch labels shape: {labels.shape}")
    print(f"  Batch input range: [{inputs.min():.2f}, {inputs.max():.2f}]")
    print(f"  Batch labels:  {labels. tolist()}")
    
    # Test model forward pass
    print("\n6. Testing model forward pass...")
    from src.models.cnn import LightweightCNN
    
    model = LightweightCNN(num_classes=config["model"]["num_classes"])
    model.eval()
    
    with torch. no_grad():
        outputs = model(inputs)
    
    print(f"  Model output shape: {outputs.shape}")
    print(f"  Model output range: [{outputs.min():.2f}, {outputs.max():.2f}]")
    print(f"  Sample outputs:\n{outputs[0]}")
    
    # Calculate loss
    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(outputs, labels)
    print(f"  Sample loss: {loss.item():.4f}")
    
    if loss.item() > 2.0:
        print("  ⚠ Loss is very high - model might not be learning properly")
    
    print("\n" + "="*60)
    print("Diagnostics complete!")
    print("="*60)


if __name__ == "__main__":
    diagnose_dataset()

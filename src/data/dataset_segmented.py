"""Dataset class for segmented ICBHI data."""

import os
import torch
from torch.utils.data import Dataset
from pathlib import Path
import random
from . preprocessing import AudioPreprocessor


class ICBHISegmentedDataset(Dataset):
    """
    Dataset class for pre-segmented ICBHI respiratory sounds.
    
    Expected directory structure:
    data/ICBHI_segmented/
    ├── normal/
    │   ├── 101_1b1_Al_sc_Meditron_seg000_normal.wav
    │   └── ... 
    ├── crackle/
    │   ├── 101_1b1_Al_sc_Meditron_seg001_crackle.wav
    │   └── ...
    ├── wheeze/
    │   └── ... 
    └── both/
        └── ...
    """
    
    CLASS_MAP = {
        'normal': 0,
        'crackle': 1,
        'wheeze': 2,
        'both': 3
    }
    
    def __init__(self, root_dir, split='train', config=None, augment=False):
        """
        Initialize dataset.
        
        Args:
            root_dir: Root directory containing segmented files
            split: Dataset split ('train', 'val', 'test')
            config:  Configuration dictionary
            augment: Whether to apply data augmentation
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.augment = augment and (split == 'train')
        
        # Initialize preprocessor
        if config:
            self.preprocessor = AudioPreprocessor(
                sample_rate=config['data']['sample_rate'],
                n_mels=config['data']['n_mels'],
                n_fft=config['data']['n_fft'],
                hop_length=config['data']['hop_length'],
                duration=config['data']['duration'],
                augment=self.augment
            )
        else:
            self.preprocessor = AudioPreprocessor(augment=self.augment)
        
        # Load data
        self.data = self._load_data()
        
        # Split data
        self._split_data(config)
    
    def _load_data(self):
        """Load all segmented audio files."""
        data = []
        
        for class_name, class_idx in self.CLASS_MAP.items():
            class_dir = self.root_dir / class_name
            
            if not class_dir.exists():
                print(f"Warning: Directory not found: {class_dir}")
                continue
            
            # Get all .  wav files in this class directory
            wav_files = list(class_dir.glob("*.wav"))
            
            for wav_file in wav_files: 
                data.append((str(wav_file), class_idx))
        
        if not data:
            raise ValueError(f"No audio files found in {self.root_dir}")
        
        # Shuffle data for consistent splits
        random.seed(42)
        random.shuffle(data)
        
        return data
    
    def _split_data(self, config):
        """Split data into train/val/test."""
        total = len(self.data)
        
        if config: 
            train_split = config['data']. get('train_split', 0.7)
            val_split = config['data'].get('val_split', 0.15)
        else:
            train_split = 0.7
            val_split = 0.15
        
        train_size = int(train_split * total)
        val_size = int(val_split * total)
        
        if self.split == 'train': 
            self.data = self.data[:train_size]
        elif self.split == 'val':
            self.data = self.data[train_size:train_size + val_size]
        else:  # test
            self.data = self.data[train_size + val_size:]
        
        print(f"Loaded {len(self.data)} samples for {self.split} split")
        
        # Print class distribution
        class_counts = {}
        for _, label in self.data:
            class_name = [k for k, v in self.CLASS_MAP.items() if v == label][0]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        print(f"Class distribution for {self.split}:")
        for class_name, count in sorted(class_counts. items()):
            print(f"  {class_name}: {count} ({100*count/len(self.data):.1f}%)")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """Get item by index."""
        audio_path, label = self.data[idx]
        
        # Preprocess audio
        mel_spec = self.preprocessor.preprocess(audio_path)
        
        return mel_spec, label


if __name__ == "__main__": 
    # Test dataset
    import yaml
    
    with open("config.yaml", "r") as f:
        config = yaml. safe_load(f)
    
    # Update config to point to segmented data
    config['data']['dataset_path'] = 'data/ICBHI_segmented'
    
    dataset = ICBHISegmentedDataset(
        root_dir=config['data']['dataset_path'],
        split='train',
        config=config,
        augment=True
    )
    
    print(f"\nDataset size: {len(dataset)}")
    mel_spec, label = dataset[0]
    print(f"Mel-spectrogram shape: {mel_spec.shape}")
    print(f"Label: {label}")

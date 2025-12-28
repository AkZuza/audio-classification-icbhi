"""Dataset class for ICBHI respiratory sounds."""

import os
import torch
from torch.utils.data import Dataset
import pandas as pd
from pathlib import Path
from . preprocessing import AudioPreprocessor


class ICBHIDataset(Dataset):
    """
    Dataset class for ICBHI respiratory sound database.

    Expected directory structure:
    data/ICBHI/
    ├── audio_and_txt_files/
    │   ├── 101_1b1_Al_sc_Meditron. wav
    │   ├── 101_1b1_Al_sc_Meditron. txt
    │   └── ...
    """

    # Class mapping
    CLASS_MAP = {"normal": 0, "crackles": 1, "wheezes": 2, "both": 3}

    def __init__(self, root_dir, split="train", config=None, augment=False):
        """
        Initialize ICBHI dataset.

        Args:
            root_dir: Root directory containing the ICBHI dataset
            split:  Dataset split ('train', 'val', 'test')
            config:  Configuration dictionary
            augment: Whether to apply data augmentation
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.augment = augment and (split == "train")

        # Initialize preprocessor
        if config: 
            self.preprocessor = AudioPreprocessor(
                sample_rate=config["data"]["sample_rate"],
                n_mels=config["data"]["n_mels"],
                n_fft=config["data"]["n_fft"],
                hop_length=config["data"]["hop_length"],
                duration=config["data"]["duration"],
                augment=self.augment,
            )
        else:
            self.preprocessor = AudioPreprocessor(augment=self.augment)

        # Load and split data
        self.data = self._load_data()

    def _load_data(self):
        """
        Load audio files and labels from the dataset. 

        Returns:
            List of (audio_path, label) tuples
        """
        data = []
        audio_dir = self.root_dir / "audio_and_txt_files"

        if not audio_dir.exists():
            raise ValueError(f"Audio directory not found: {audio_dir}")

        # Get all . wav files
        wav_files = sorted(audio_dir.glob("*. wav"))

        for wav_file in wav_files:
            txt_file = wav_file.with_suffix(".txt")

            if txt_file.exists():
                # Read annotation file
                label = self._parse_annotation(txt_file)
                data.append((str(wav_file), label))

        # Split data
        total = len(data)
        train_size = int(0.7 * total)
        val_size = int(0.15 * total)

        if self.split == "train": 
            data = data[:train_size]
        elif self.split == "val": 
            data = data[train_size :  train_size + val_size]
        else:  # test
            data = data[train_size + val_size :]

        print(f"Loaded {len(data)} samples for {self.split} split")
        return data

    def _parse_annotation(self, txt_file):
        """
        Parse annotation file to extract label. 

        Annotation format:  start, end, crackles, wheezes

        Returns:
            Label index (0: normal, 1: crackles, 2: wheezes, 3: both)
        """
        with open(txt_file, "r") as f:
            lines = f.readlines()

        # Aggregate labels from all cycles
        has_crackles = False
        has_wheezes = False

        for line in lines:
            parts = line.strip().split("\t")
            if len(parts) >= 4:
                crackle = int(parts[2])
                wheeze = int(parts[3])

                if crackle == 1:
                    has_crackles = True
                if wheeze == 1:
                    has_wheezes = True

        # Determine label
        if has_crackles and has_wheezes:
            return self.CLASS_MAP["both"]
        elif has_crackles: 
            return self.CLASS_MAP["crackles"]
        elif has_wheezes:
            return self.CLASS_MAP["wheezes"]
        else: 
            return self.CLASS_MAP["normal"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Get item by index.

        Returns:
            Tuple of (mel_spectrogram, label)
        """
        audio_path, label = self.data[idx]

        # Preprocess audio
        mel_spec = self.preprocessor.preprocess(audio_path)

        return mel_spec, label


if __name__ == "__main__": 
    # Test dataset
    import yaml

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    dataset = ICBHIDataset(
        root_dir="data/ICBHI", split="train", config=config, augment=True
    )

    print(f"Dataset size: {len(dataset)}")
    mel_spec, label = dataset[0]
    print(f"Mel-spectrogram shape: {mel_spec.shape}")
    print(f"Label: {label}")

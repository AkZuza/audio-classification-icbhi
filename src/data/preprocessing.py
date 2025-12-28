"""Audio preprocessing utilities."""

import torch
import torchaudio
import torchaudio.transforms as T
import numpy as np


class AudioPreprocessor:
    """
    Audio preprocessing for respiratory sounds. 

    Handles:
    - Resampling
    - Mel-spectrogram conversion
    - Normalization
    - Augmentation
    """

    def __init__(
        self,
        sample_rate=16000,
        n_mels=128,
        n_fft=2048,
        hop_length=512,
        duration=5.0,
        augment=False,
    ):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.duration = duration
        self. augment = augment
        self.target_length = int(sample_rate * duration)

        # Mel-spectrogram transform
        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2.0,
        )

        # Amplitude to dB
        self.amplitude_to_db = T.AmplitudeToDB()

        # Augmentation transforms
        if augment:
            self.time_stretch = T.TimeStretch(fixed_rate=None)
            self.freq_mask = T.FrequencyMasking(freq_mask_param=15)
            self.time_mask = T.TimeMasking(time_mask_param=35)

    def load_audio(self, audio_path):
        """Load audio file and resample if necessary."""
        waveform, sr = torchaudio.load(audio_path)

        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Resample if necessary
        if sr != self.sample_rate:
            resampler = T.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)

        return waveform

    def pad_or_crop(self, waveform):
        """Pad or crop waveform to target length."""
        current_length = waveform.shape[1]

        if current_length < self.target_length:
            # Pad with zeros
            padding = self.target_length - current_length
            waveform = torch. nn.functional.pad(waveform, (0, padding))
        elif current_length > self.target_length:
            # Crop from center
            start = (current_length - self. target_length) // 2
            waveform = waveform[:, start :  start + self.target_length]

        return waveform

    def add_noise(self, waveform, noise_factor=0.005):
        """Add Gaussian noise to waveform."""
        noise = torch.randn_like(waveform) * noise_factor
        return waveform + noise

    def time_shift(self, waveform, shift_max=0.2):
        """Randomly shift waveform in time."""
        shift = int(np.random.uniform(-shift_max, shift_max) * waveform.shape[1])
        return torch.roll(waveform, shift, dims=1)

    def augment_waveform(self, waveform):
        """Apply random augmentations to waveform."""
        if np.random.random() > 0.5:
            waveform = self.add_noise(waveform)

        if np.random.random() > 0.5:
            waveform = self.time_shift(waveform)

        return waveform

    def augment_spectrogram(self, mel_spec):
        """Apply SpecAugment to mel-spectrogram."""
        mel_spec = self.freq_mask(mel_spec)
        mel_spec = self.time_mask(mel_spec)
        return mel_spec

    def normalize(self, mel_spec):
        """Normalize mel-spectrogram."""
        # Normalize to zero mean and unit variance
        mean = mel_spec.mean()
        std = mel_spec.std()
        return (mel_spec - mean) / (std + 1e-8)

    def preprocess(self, audio_path):
        """
        Complete preprocessing pipeline.

        Args:
            audio_path: Path to audio file

        Returns:
            Preprocessed mel-spectrogram tensor
        """
        # Load audio
        waveform = self. load_audio(audio_path)

        # Pad or crop to target length
        waveform = self.pad_or_crop(waveform)

        # Apply waveform augmentation (if enabled and training)
        if self.augment: 
            waveform = self. augment_waveform(waveform)

        # Convert to mel-spectrogram
        mel_spec = self.mel_spectrogram(waveform)

        # Convert to dB scale
        mel_spec = self. amplitude_to_db(mel_spec)

        # Apply spectrogram augmentation (if enabled and training)
        if self.augment:
            mel_spec = self.augment_spectrogram(mel_spec)

        # Normalize
        mel_spec = self.normalize(mel_spec)

        return mel_spec


if __name__ == "__main__": 
    # Test preprocessing
    preprocessor = AudioPreprocessor(augment=True)
    print(f"Target audio length: {preprocessor.target_length} samples")
    print(f"Sample rate: {preprocessor.sample_rate} Hz")
    print(f"Mel bins: {preprocessor.n_mels}")

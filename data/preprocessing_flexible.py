"""Flexible audio preprocessor that adapts to different durations."""

import torch
import torchaudio
import torchaudio.transforms as T
import numpy as np


class FlexibleAudioPreprocessor: 
    """
    Audio preprocessor that handles variable segment durations.
    """

    def __init__(
        self,
        sample_rate=16000,
        n_mels=128,
        n_fft=2048,
        hop_length=512,
        duration=5.0,
        augment=False,
        min_duration=0.5,
    ):
        self.sample_rate = sample_rate
        self. n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.duration = duration
        self.augment = augment
        self.min_duration = min_duration
        self.target_length = int(sample_rate * duration)

        # Adjust n_fft and hop_length for short segments
        if duration < 1.0:
            self. n_fft = min(1024, int(sample_rate * duration / 2))
            self.hop_length = self.n_fft // 4

        # Mel-spectrogram transform
        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
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
            start = (current_length - self.target_length) // 2
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
        mean = mel_spec.mean()
        std = mel_spec. std()
        return (mel_spec - mean) / (std + 1e-8)

    def resize_spectrogram(self, mel_spec, target_time_steps=None):
        """
        Resize spectrogram to target time steps using interpolation.
        
        Args:
            mel_spec: Input spectrogram (C, H, W)
            target_time_steps: Target width (time dimension)
        
        Returns:
            Resized spectrogram
        """
        if target_time_steps is None:
            # Calculate expected time steps for the duration
            target_time_steps = int(np.ceil((self.target_length / self.hop_length)))
        
        # Ensure minimum size
        target_time_steps = max(target_time_steps, 32)
        
        if mel_spec.shape[-1] != target_time_steps:
            # Add batch dimension if not present
            if mel_spec.dim() == 2:
                mel_spec = mel_spec.unsqueeze(0).unsqueeze(0)
            elif mel_spec.dim() == 3:
                mel_spec = mel_spec.unsqueeze(0)
            
            # Resize using interpolation
            mel_spec = torch.nn.functional.interpolate(
                mel_spec,
                size=(self.n_mels, target_time_steps),
                mode='bilinear',
                align_corners=False
            )
            
            # Remove batch dimension
            mel_spec = mel_spec.squeeze(0)
        
        return mel_spec

    def preprocess(self, audio_path):
        """
        Complete preprocessing pipeline. 

        Args:
            audio_path: Path to audio file

        Returns:
            Preprocessed mel-spectrogram tensor
        """
        # Load audio
        waveform = self.load_audio(audio_path)

        # Pad or crop to target length
        waveform = self.pad_or_crop(waveform)

        # Apply waveform augmentation (if enabled and training)
        if self.augment: 
            waveform = self. augment_waveform(waveform)

        # Convert to mel-spectrogram
        mel_spec = self.mel_spectrogram(waveform)

        # Convert to dB scale
        mel_spec = self. amplitude_to_db(mel_spec)

        # Resize to consistent size
        mel_spec = self.resize_spectrogram(mel_spec)

        # Apply spectrogram augmentation (if enabled and training)
        if self.augment:
            mel_spec = self.augment_spectrogram(mel_spec)

        # Normalize
        mel_spec = self.normalize(mel_spec)

        return mel_spec
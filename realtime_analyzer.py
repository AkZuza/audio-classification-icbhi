"""
Real-time respiratory sound analyzer with parallel processing and visualization. 

This script: 
1. Loads a . wav file (max 15 seconds)
2. Segments it into small windows (e.g., 1 second with overlap)
3. Processes segments in parallel using multiple model instances
4. Displays results as a timeline graph with crackles (purple) and wheezes (green)
"""

import torch
import torch.multiprocessing as mp
import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pathlib import Path
import argparse
from tqdm import tqdm
from dataclasses import dataclass
from typing import List, Tuple
import warnings
warnings.filterwarnings('ignore')

from src.models.cnn import LightweightCNN
from src.models. resnet import CompactResNet
from src.data.preprocessing import AudioPreprocessor


@dataclass
class SegmentResult:
    """Result for a single audio segment."""
    start_time: float
    end_time: float
    has_crackle: bool
    has_wheeze: bool
    crackle_confidence: float
    wheeze_confidence: float
    normal_confidence: float
    both_confidence: float
    predicted_class: str


class ParallelAudioAnalyzer: 
    """Analyze respiratory sounds with parallel CNN processing."""
    
    def __init__(
        self,
        model_path: str,
        segment_duration: float = 1.0,
        overlap: float = 0.5,
        sample_rate: int = 16000,
        n_workers: int = 4,
        device: str = 'cuda'
    ):
        """
        Initialize analyzer. 
        
        Args:
            model_path: Path to trained model checkpoint
            segment_duration: Duration of each segment in seconds
            overlap:  Overlap between segments (0-1)
            sample_rate: Target sample rate
            n_workers:  Number of parallel workers
            device:  Device to use ('cuda' or 'cpu')
        """
        self.model_path = model_path
        self.segment_duration = segment_duration
        self.overlap = overlap
        self.sample_rate = sample_rate
        self.n_workers = n_workers
        self.device = device
        
        # Load model and config
        self.model, self.config = self._load_model()
        
        # Initialize preprocessor
        self.preprocessor = AudioPreprocessor(
            sample_rate=sample_rate,
            n_mels=self.config['data']['n_mels'],
            n_fft=self.config['data']['n_fft'],
            hop_length=self.config['data']['hop_length'],
            duration=segment_duration,
            augment=False
        )
        
        # Class mapping
        self.class_names = self.config['classes']
        self.class_map = {i: name for i, name in enumerate(self.class_names)}
    
    def _load_model(self):
        """Load trained model from checkpoint."""
        print(f"Loading model from {self.model_path}...")
        
        checkpoint = torch.load(self.model_path, map_location='cpu', weights_only=False)
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
        model.eval()
        
        print(f"✓ Model loaded (architecture: {config['model']['architecture']})")
        
        return model, config
    
    def load_audio(self, audio_path: str, max_duration: float = 15.0) -> np.ndarray:
        """
        Load audio file.
        
        Args:
            audio_path: Path to audio file
            max_duration: Maximum duration in seconds
        
        Returns:
            Audio waveform as numpy array
        """
        print(f"\nLoading audio:  {audio_path}")
        
        # Load audio
        audio, sr = librosa.load(audio_path, sr=self.sample_rate, duration=max_duration)
        
        duration = len(audio) / self.sample_rate
        print(f"✓ Audio loaded:  {duration:.2f}s, {sr}Hz")
        
        return audio
    
    def segment_audio(self, audio: np.ndarray) -> List[Tuple[np.ndarray, float, float]]:
        """
        Segment audio into overlapping windows.
        
        Args:
            audio: Audio waveform
        
        Returns: 
            List of (segment, start_time, end_time) tuples
        """
        duration = len(audio) / self.sample_rate
        segment_samples = int(self.segment_duration * self.sample_rate)
        hop_samples = int(segment_samples * (1 - self.overlap))
        
        segments = []
        start_sample = 0
        
        while start_sample + segment_samples <= len(audio):
            end_sample = start_sample + segment_samples
            segment = audio[start_sample:end_sample]
            
            start_time = start_sample / self.sample_rate
            end_time = end_sample / self.sample_rate
            
            segments.append((segment, start_time, end_time))
            
            start_sample += hop_samples
        
        # Handle last segment if there's remaining audio
        if start_sample < len(audio):
            segment = audio[start_sample:]
            # Pad if necessary
            if len(segment) < segment_samples:
                segment = np.pad(segment, (0, segment_samples - len(segment)))
            
            start_time = start_sample / self.sample_rate
            end_time = duration
            segments.append((segment, start_time, end_time))
        
        print(f"✓ Created {len(segments)} segments ({self.segment_duration}s each, {self.overlap*100:.0f}% overlap)")
        
        return segments
    
    def process_segment(self, segment_data:  Tuple[np.ndarray, float, float]) -> SegmentResult:
        """
        Process a single segment. 
        
        Args:
            segment_data:  Tuple of (segment, start_time, end_time)
        
        Returns: 
            SegmentResult
        """
        segment, start_time, end_time = segment_data
        
        # Save segment temporarily
        temp_path = Path('/tmp/temp_segment.wav')
        sf.write(temp_path, segment, self.sample_rate)
        
        # Preprocess
        mel_spec = self.preprocessor.preprocess(str(temp_path))
        mel_spec = mel_spec.unsqueeze(0)  # Add batch dimension
        
        # Move to device
        device = torch.device(self.device if torch.cuda.is_available() else 'cpu')
        mel_spec = mel_spec.to(device)
        model = self.model.to(device)
        
        # Inference
        with torch.no_grad():
            outputs = model(mel_spec)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class_idx = outputs.argmax(dim=1).item()
        
        # Extract probabilities
        probs = probabilities[0]. cpu().numpy()
        
        # Map to class names
        normal_conf = probs[0]
        crackle_conf = probs[1]
        wheeze_conf = probs[2]
        both_conf = probs[3]
        
        # Determine if crackles or wheezes are present
        # Consider both "crackle" and "both" classes for crackle detection
        # Consider both "wheeze" and "both" classes for wheeze detection
        has_crackle = (crackle_conf > 0.5) or (both_conf > 0.5)
        has_wheeze = (wheeze_conf > 0.5) or (both_conf > 0.5)
        
        # Aggregate confidences
        total_crackle_conf = crackle_conf + both_conf
        total_wheeze_conf = wheeze_conf + both_conf
        
        predicted_class = self.class_map[predicted_class_idx]
        
        return SegmentResult(
            start_time=start_time,
            end_time=end_time,
            has_crackle=has_crackle,
            has_wheeze=has_wheeze,
            crackle_confidence=total_crackle_conf,
            wheeze_confidence=total_wheeze_conf,
            normal_confidence=normal_conf,
            both_confidence=both_conf,
            predicted_class=predicted_class
        )
    
    def analyze_audio(self, audio_path: str) -> List[SegmentResult]:
        """
        Analyze complete audio file.
        
        Args:
            audio_path: Path to audio file
        
        Returns:
            List of SegmentResult objects
        """
        # Load audio
        audio = self. load_audio(audio_path)
        
        # Segment audio
        segments = self.segment_audio(audio)
        
        # Process segments
        print(f"\nProcessing {len(segments)} segments...")
        results = []
        
        # Sequential processing with progress bar
        for segment_data in tqdm(segments, desc="Analyzing segments"):
            result = self. process_segment(segment_data)
            results.append(result)
        
        print(f"✓ Analysis complete!")
        
        return results, audio
    
    def visualize_results(
        self,
        results: List[SegmentResult],
        audio:  np.ndarray,
        save_path: str = None,
        show:  bool = True
    ):
        """
        Visualize analysis results as a timeline. 
        
        Args:
            results: List of SegmentResult objects
            audio: Original audio waveform
            save_path:  Path to save figure
            show: Whether to display the plot
        """
        fig, axes = plt.subplots(3, 1, figsize=(16, 10))
        
        # Time axis
        duration = len(audio) / self.sample_rate
        time_axis = np.linspace(0, duration, len(audio))
        
        # 1. Waveform
        ax1 = axes[0]
        ax1.plot(time_axis, audio, color='gray', linewidth=0.5, alpha=0.7)
        ax1.set_ylabel('Amplitude', fontsize=12)
        ax1.set_title('Audio Waveform', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim([0, duration])
        
        # 2. Detection Timeline (Bar chart style)
        ax2 = axes[1]
        
        crackle_times = []
        crackle_confidences = []
        wheeze_times = []
        wheeze_confidences = []
        
        for result in results:
            mid_time = (result.start_time + result. end_time) / 2
            
            if result.has_crackle:
                crackle_times.append(mid_time)
                crackle_confidences.append(result.crackle_confidence)
            
            if result.has_wheeze:
                wheeze_times. append(mid_time)
                wheeze_confidences.append(result.wheeze_confidence)
        
        # Plot as scatter with vertical lines
        for i, result in enumerate(results):
            mid_time = (result.start_time + result.end_time) / 2
            
            # Crackles (purple)
            if result.has_crackle:
                ax2.vlines(mid_time, 0, result.crackle_confidence, 
                          colors='purple', linewidth=3, alpha=0.7, label='Crackle' if i == 0 else '')
            
            # Wheezes (green)
            if result.has_wheeze:
                ax2.vlines(mid_time, 0, result.wheeze_confidence, 
                          colors='green', linewidth=3, alpha=0.7, label='Wheeze' if i == 0 else '')
        
        ax2.set_ylabel('Confidence', fontsize=12)
        ax2.set_title('Respiratory Sound Detection (Purple=Crackles, Green=Wheezes)', 
                     fontsize=14, fontweight='bold')
        ax2.set_ylim([0, 1.0])
        ax2.set_xlim([0, duration])
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper right', fontsize=10)
        
        # 3. Continuous confidence timeline
        ax3 = axes[2]
        
        # Create continuous timeline
        times = [result.start_time for result in results]
        crackle_confs = [result.crackle_confidence for result in results]
        wheeze_confs = [result.wheeze_confidence for result in results]
        
        ax3.plot(times, crackle_confs, color='purple', linewidth=2, 
                marker='o', markersize=4, label='Crackles', alpha=0.8)
        ax3.plot(times, wheeze_confs, color='green', linewidth=2, 
                marker='o', markersize=4, label='Wheezes', alpha=0.8)
        ax3.fill_between(times, crackle_confs, alpha=0.2, color='purple')
        ax3.fill_between(times, wheeze_confs, alpha=0.2, color='green')
        
        # Add threshold line
        ax3.axhline(y=0.5, color='red', linestyle='--', linewidth=1, 
                   alpha=0.5, label='Detection Threshold')
        
        ax3.set_xlabel('Time (seconds)', fontsize=12)
        ax3.set_ylabel('Confidence', fontsize=12)
        ax3.set_title('Confidence Timeline', fontsize=14, fontweight='bold')
        ax3.set_ylim([0, 1.0])
        ax3.set_xlim([0, duration])
        ax3.grid(True, alpha=0.3)
        ax3.legend(loc='upper right', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\n✓ Visualization saved to:  {save_path}")
        
        if show:
            plt. show()
        else:
            plt.close()
    
    def print_summary(self, results: List[SegmentResult]):
        """
        Print analysis summary.
        
        Args:
            results: List of SegmentResult objects
        """
        total_segments = len(results)
        crackle_segments = sum(1 for r in results if r.has_crackle)
        wheeze_segments = sum(1 for r in results if r.has_wheeze)
        both_segments = sum(1 for r in results if r.has_crackle and r.has_wheeze)
        normal_segments = sum(1 for r in results if not r.has_crackle and not r.has_wheeze)
        
        print("\n" + "=" * 70)
        print("ANALYSIS SUMMARY")
        print("=" * 70)
        print(f"Total segments analyzed: {total_segments}")
        print(f"Normal segments: {normal_segments} ({100*normal_segments/total_segments:.1f}%)")
        print(f"Crackle detections: {crackle_segments} ({100*crackle_segments/total_segments:.1f}%)")
        print(f"Wheeze detections: {wheeze_segments} ({100*wheeze_segments/total_segments:.1f}%)")
        print(f"Both detected: {both_segments} ({100*both_segments/total_segments:.1f}%)")
        
        # Time ranges
        if crackle_segments > 0:
            crackle_times = [(r.start_time, r. end_time) for r in results if r.has_crackle]
            print(f"\nCrackle time ranges:")
            for start, end in crackle_times[: 5]:  # Show first 5
                print(f"  {start:.2f}s - {end:.2f}s")
            if len(crackle_times) > 5:
                print(f"  ... and {len(crackle_times) - 5} more")
        
        if wheeze_segments > 0:
            wheeze_times = [(r.start_time, r.end_time) for r in results if r.has_wheeze]
            print(f"\nWheeze time ranges:")
            for start, end in wheeze_times[:5]: 
                print(f"  {start:.2f}s - {end:.2f}s")
            if len(wheeze_times) > 5:
                print(f"  ... and {len(wheeze_times) - 5} more")
        
        print("=" * 70)
    
    def export_results(self, results: List[SegmentResult], output_path: str):
        """
        Export results to CSV.
        
        Args:
            results: List of SegmentResult objects
            output_path: Path to save CSV file
        """
        import csv
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Start Time (s)',
                'End Time (s)',
                'Has Crackle',
                'Has Wheeze',
                'Crackle Confidence',
                'Wheeze Confidence',
                'Normal Confidence',
                'Both Confidence',
                'Predicted Class'
            ])
            
            for result in results:
                writer.writerow([
                    f"{result.start_time:.3f}",
                    f"{result.end_time:.3f}",
                    result.has_crackle,
                    result.has_wheeze,
                    f"{result.crackle_confidence:.4f}",
                    f"{result.wheeze_confidence:.4f}",
                    f"{result.normal_confidence:.4f}",
                    f"{result.both_confidence:.4f}",
                    result.predicted_class
                ])
        
        print(f"\n✓ Results exported to: {output_path}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Real-time respiratory sound analyzer with parallel processing"
    )
    parser.add_argument(
        "--audio",
        type=str,
        required=True,
        help="Path to audio file (max 15 seconds)"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--segment-duration",
        type=float,
        default=1.0,
        help="Duration of each segment in seconds (default: 1.0)"
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.5,
        help="Overlap between segments (0-1, default: 0.5)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="analysis_results",
        help="Directory to save results"
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
        help="Don't display the plot"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Create analyzer
    analyzer = ParallelAudioAnalyzer(
        model_path=args.model,
        segment_duration=args.segment_duration,
        overlap=args.overlap,
        device=args.device
    )
    
    # Analyze audio
    results, audio = analyzer. analyze_audio(args.audio)
    
    # Print summary
    analyzer.print_summary(results)
    
    # Visualize results
    audio_name = Path(args.audio).stem
    viz_path = output_dir / f"{audio_name}_analysis.png"
    analyzer.visualize_results(
        results,
        audio,
        save_path=str(viz_path),
        show=not args.no_display
    )
    
    # Export results
    csv_path = output_dir / f"{audio_name}_results.csv"
    analyzer.export_results(results, str(csv_path))
    
    print(f"\n✓ All results saved to: {output_dir}")


if __name__ == "__main__":
    main()

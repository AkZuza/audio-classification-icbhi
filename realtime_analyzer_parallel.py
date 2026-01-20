"""
Analyzer with configurable detection threshold.
"""

import torch
import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from tqdm import tqdm
from dataclasses import dataclass
from typing import List, Tuple
import warnings
warnings.filterwarnings('ignore')

from src.models.cnn import LightweightCNN
from src.models.resnet import CompactResNet
from data.preprocessing_flexible import FlexibleAudioPreprocessor


@dataclass
class SegmentResult:
    """Result for a single audio segment."""
    start_time: float
    end_time: float
    has_crackle:  bool
    has_wheeze:  bool
    crackle_confidence: float
    wheeze_confidence: float
    normal_confidence:  float
    both_confidence: float
    predicted_class: str


class ConfigurableAudioAnalyzer:
    """Analyzer with configurable detection threshold."""
    
    def __init__(
        self,
        model_path: str,
        segment_duration: float = 1.0,
        overlap: float = 0.5,
        sample_rate: int = 16000,
        device: str = 'cuda',
        crackle_threshold: float = 0.3,
        wheeze_threshold: float = 0.3,
    ):
        """
        Initialize analyzer. 
        
        Args:
            model_path: Path to trained model checkpoint
            segment_duration: Duration of each segment in seconds
            overlap:  Overlap between segments (0-1)
            sample_rate:  Target sample rate
            device: Device to use
            crackle_threshold: Detection threshold for crackles (0-1)
            wheeze_threshold: Detection threshold for wheezes (0-1)
        """
        self.model_path = model_path
        self. segment_duration = segment_duration
        self.overlap = overlap
        self.sample_rate = sample_rate
        self.device = device
        self.crackle_threshold = crackle_threshold
        self.wheeze_threshold = wheeze_threshold
        
        # Load model and config
        self.model, self.config = self._load_model()
        
        # Initialize preprocessor
        self.preprocessor = FlexibleAudioPreprocessor(
            sample_rate=sample_rate,
            n_mels=self.config['data']['n_mels'],
            n_fft=min(2048, int(sample_rate * segment_duration / 2)),
            hop_length=256 if segment_duration < 1.0 else 512,
            duration=segment_duration,
            augment=False
        )
        
        # Class mapping
        self.class_names = self. config['classes']
        self.class_map = {i: name for i, name in enumerate(self.class_names)}
        
        print(f"\n⚙️  Detection Thresholds:")
        print(f"   Crackle threshold: {self.crackle_threshold:.2f}")
        print(f"   Wheeze threshold:   {self.wheeze_threshold:.2f}")
        
        if self.crackle_threshold < 0.3 or self.wheeze_threshold < 0.3:
            print(f"   ⚠️  Low threshold - High sensitivity (more detections, possibly more false positives)")
        elif self.crackle_threshold > 0.6 or self.wheeze_threshold > 0.6:
            print(f"   ⚠️  High threshold - High specificity (fewer detections, possibly missing some)")
        else:
            print(f"   ✓ Balanced threshold")
    
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
        
        print(f"✓ Model loaded")
        
        return model, config
    
    def load_audio(self, audio_path: str, max_duration: float = 15.0) -> np.ndarray:
        """Load audio file."""
        print(f"\nLoading audio:  {audio_path}")
        audio, sr = librosa.load(audio_path, sr=self. sample_rate, duration=max_duration)
        duration = len(audio) / self.sample_rate
        print(f"✓ Audio loaded:  {duration:.2f}s")
        return audio
    
    def segment_audio(self, audio: np.ndarray) -> List[Tuple[np.ndarray, float, float]]:
        """Segment audio into overlapping windows."""
        duration = len(audio) / self.sample_rate
        segment_samples = int(self.segment_duration * self.sample_rate)
        hop_samples = int(segment_samples * (1 - self.overlap))
        
        segments = []
        start_sample = 0
        
        while start_sample + segment_samples <= len(audio):
            end_sample = start_sample + segment_samples
            segment = audio[start_sample:end_sample]
            start_time = start_sample / self.sample_rate
            end_time = end_sample / self. sample_rate
            segments.append((segment, start_time, end_time))
            start_sample += hop_samples
        
        # Last segment
        if start_sample < len(audio):
            segment = audio[start_sample:]
            if len(segment) < segment_samples:
                segment = np.pad(segment, (0, segment_samples - len(segment)))
            start_time = start_sample / self.sample_rate
            end_time = duration
            segments.append((segment, start_time, end_time))
        
        print(f"✓ Created {len(segments)} segments")
        return segments
    
    def process_segments_batch(self, segments: List, batch_size: int = 32) -> List[SegmentResult]: 
        """Process segments in batches."""
        device = torch.device(self.device if torch.cuda.is_available() else 'cpu')
        self.model = self.model. to(device)
        self.model.eval()
        
        results = []
        
        # Preprocess
        print("Preprocessing segments...")
        preprocessed = []
        segment_info = []
        
        temp_dir = Path('/tmp/audio_segments')
        temp_dir.mkdir(exist_ok=True)
        
        for idx, (segment, start_time, end_time) in enumerate(tqdm(segments, desc="Preprocessing")):
            temp_path = temp_dir / f'segment_{idx:04d}.wav'
            sf. write(temp_path, segment, self.sample_rate)
            
            try:
                mel_spec = self.preprocessor. preprocess(str(temp_path))
                preprocessed.append(mel_spec)
                segment_info.append((start_time, end_time))
            except Exception as e:
                print(f"Warning: Failed segment {idx}: {e}")
            finally:
                if temp_path.exists():
                    temp_path.unlink()
        
        # Batch inference
        print(f"Processing {len(preprocessed)} segments...")
        
        with torch.no_grad():
            for i in tqdm(range(0, len(preprocessed), batch_size), desc="Inference"):
                batch_specs = preprocessed[i:i + batch_size]
                batch_info = segment_info[i:i + batch_size]
                
                try:
                    batch_tensor = torch.stack(batch_specs).to(device)
                    outputs = self.model(batch_tensor)
                    probabilities = torch.softmax(outputs, dim=1)
                    predicted_classes = outputs.argmax(dim=1)
                    
                    for j, (start_time, end_time) in enumerate(batch_info):
                        probs = probabilities[j].cpu().numpy()
                        predicted_class_idx = predicted_classes[j]. item()
                        
                        # Extract confidences
                        normal_conf = float(probs[0])
                        crackle_conf = float(probs[1])
                        wheeze_conf = float(probs[2])
                        both_conf = float(probs[3])
                        
                        # Apply thresholds (KEY PART!)
                        total_crackle_conf = min(crackle_conf + both_conf, 1.0)
                        total_wheeze_conf = min(wheeze_conf + both_conf, 1.0)
                        
                        has_crackle = total_crackle_conf > self.crackle_threshold
                        has_wheeze = total_wheeze_conf > self.wheeze_threshold
                        
                        predicted_class = self.class_map[predicted_class_idx]
                        
                        result = SegmentResult(
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
                        
                        results.append(result)
                
                except Exception as e:
                    print(f"Warning:  Batch failed: {e}")
        
        # Cleanup
        try:
            temp_dir.rmdir()
        except:
            pass
        
        return results
    
    def analyze_audio(self, audio_path: str) -> tuple:
        """Analyze audio file."""
        audio = self.load_audio(audio_path)
        segments = self.segment_audio(audio)
        results = self.process_segments_batch(segments)
        print(f"✓ Analysis complete!")
        return results, audio
    
    def visualize_results(self, results: List[SegmentResult], audio: np.ndarray, 
                         save_path: str = None, show: bool = True):
        """Visualize results with threshold line."""
        fig, axes = plt.subplots(3, 1, figsize=(16, 10))
        
        duration = len(audio) / self.sample_rate
        time_axis = np.linspace(0, duration, len(audio))
        
        # Waveform
        ax1 = axes[0]
        ax1.plot(time_axis, audio, color='gray', linewidth=0.5, alpha=0.7)
        ax1.set_ylabel('Amplitude', fontsize=12)
        ax1.set_title('Audio Waveform', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim([0, duration])
        
        # Detection bars
        ax2 = axes[1]
        
        for i, result in enumerate(results):
            mid_time = (result.start_time + result. end_time) / 2
            
            if result.has_crackle:
                ax2.vlines(mid_time, 0, result.crackle_confidence, 
                          colors='purple', linewidth=4, alpha=0.7, 
                          label='Crackle' if i == 0 else '')
            
            if result.has_wheeze:
                ax2.vlines(mid_time, 0, result.wheeze_confidence, 
                          colors='green', linewidth=4, alpha=0.7, 
                          label='Wheeze' if i == 0 else '')
        
        # Add threshold lines
        ax2.axhline(y=self.crackle_threshold, color='purple', linestyle='--', 
                   linewidth=2, alpha=0.5, label=f'Crackle Threshold ({self.crackle_threshold:.2f})')
        ax2.axhline(y=self.wheeze_threshold, color='green', linestyle='--', 
                   linewidth=2, alpha=0.5, label=f'Wheeze Threshold ({self.wheeze_threshold:.2f})')
        
        ax2.set_ylabel('Confidence', fontsize=12)
        ax2.set_title('Respiratory Sound Detection (Purple=Crackles, Green=Wheezes)', 
                     fontsize=14, fontweight='bold')
        ax2.set_ylim([0, 1.0])
        ax2.set_xlim([0, duration])
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper right', fontsize=9)
        
        # Confidence timeline
        ax3 = axes[2]
        
        times = [(r.start_time + r.end_time) / 2 for r in results]
        crackle_confs = [r.crackle_confidence for r in results]
        wheeze_confs = [r.wheeze_confidence for r in results]
        
        ax3.plot(times, crackle_confs, color='purple', linewidth=2, 
                marker='o', markersize=5, label='Crackles', alpha=0.8)
        ax3.plot(times, wheeze_confs, color='green', linewidth=2, 
                marker='o', markersize=5, label='Wheezes', alpha=0.8)
        ax3.fill_between(times, crackle_confs, alpha=0.2, color='purple')
        ax3.fill_between(times, wheeze_confs, alpha=0.2, color='green')
        
        # Threshold lines
        ax3.axhline(y=self.crackle_threshold, color='purple', linestyle='--', 
                   linewidth=1, alpha=0.5)
        ax3.axhline(y=self.wheeze_threshold, color='green', linestyle='--', 
                   linewidth=1, alpha=0.5)
        
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
        """Print analysis summary."""
        total = len(results)
        crackle = sum(1 for r in results if r.has_crackle)
        wheeze = sum(1 for r in results if r.has_wheeze)
        both = sum(1 for r in results if r.has_crackle and r.has_wheeze)
        normal = sum(1 for r in results if not r.has_crackle and not r.has_wheeze)
        
        # Average confidences
        avg_crackle = np.mean([r.crackle_confidence for r in results if r.has_crackle]) if crackle > 0 else 0
        avg_wheeze = np.mean([r.wheeze_confidence for r in results if r.has_wheeze]) if wheeze > 0 else 0
        
        print("\n" + "=" * 70)
        print("ANALYSIS SUMMARY")
        print("=" * 70)
        print(f"Detection Thresholds:  Crackle={self.crackle_threshold:.2f}, Wheeze={self.wheeze_threshold:.2f}")
        print(f"\nTotal segments: {total}")
        print(f"Normal:   {normal} ({100*normal/total:.1f}%)")
        print(f"Crackle: {crackle} ({100*crackle/total:.1f}%) - Avg confidence: {avg_crackle:.2f}")
        print(f"Wheeze:   {wheeze} ({100*wheeze/total:.1f}%) - Avg confidence: {avg_wheeze:.2f}")
        print(f"Both:    {both} ({100*both/total:.1f}%)")
        print("=" * 70)
    
    def export_results(self, results: List[SegmentResult], output_path: str):
        """Export to CSV."""
        import csv
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Start (s)', 'End (s)', 'Crackle', 'Wheeze',
                'Crackle Conf', 'Wheeze Conf', 'Class'
            ])
            
            for r in results:
                writer.writerow([
                    f"{r.start_time:.3f}", f"{r.end_time:.3f}",
                    r.has_crackle, r.has_wheeze,
                    f"{r.crackle_confidence:.4f}", f"{r.wheeze_confidence:.4f}",
                    r.predicted_class
                ])
        
        print(f"✓ Results exported to: {output_path}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Respiratory sound analyzer with configurable thresholds"
    )
    parser.add_argument("--audio", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--segment-duration", type=float, default=1.0)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--crackle-threshold", type=float, default=0.3,
                       help="Detection threshold for crackles (0-1, default: 0.3)")
    parser.add_argument("--wheeze-threshold", type=float, default=0.3,
                       help="Detection threshold for wheezes (0-1, default: 0.3)")
    parser.add_argument("--output-dir", type=str, default="analysis_results")
    parser.add_argument("--device", type=str, choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument("--no-display", action='store_true')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Create analyzer with custom thresholds
    analyzer = ConfigurableAudioAnalyzer(
        model_path=args.model,
        segment_duration=args.segment_duration,
        overlap=args.overlap,
        device=args.device,
        crackle_threshold=args.crackle_threshold,
        wheeze_threshold=args.wheeze_threshold
    )
    
    # Analyze
    results, audio = analyzer.analyze_audio(args. audio)
    analyzer.print_summary(results)
    
    # Visualize
    audio_name = Path(args.audio).stem
    viz_path = output_dir / f"{audio_name}_analysis_t{args.crackle_threshold:.2f}.png"
    analyzer.visualize_results(results, audio, save_path=str(viz_path), 
                              show=not args.no_display)
    
    # Export
    csv_path = output_dir / f"{audio_name}_results_t{args.crackle_threshold:.2f}.csv"
    analyzer.export_results(results, str(csv_path))
    
    print(f"\n✓ Results saved to: {output_dir}")


if __name__ == "__main__":
    main()

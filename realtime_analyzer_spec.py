"""
Analyzer with mel-spectrogram visualization. 
Shows: 
1.  Waveform
2. Mel-spectrogram
3. Linear timeline with colored blocks
"""

import torch
import numpy as np
import librosa
import librosa.display
import soundfile as sf
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
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
    has_crackle: bool
    has_wheeze: bool
    crackle_confidence: float
    wheeze_confidence: float
    normal_confidence: float
    both_confidence: float
    predicted_class: str


class SpectrogramAudioAnalyzer: 
    """Analyzer with mel-spectrogram visualization."""
    
    def __init__(
        self,
        model_path: str,
        segment_duration: float = 1.0,
        overlap: float = 0.5,
        sample_rate: int = 16000,
        device: str = 'cuda',
        crackle_threshold:  float = 0.3,
        wheeze_threshold: float = 0.3,
    ):
        """Initialize analyzer."""
        self.model_path = model_path
        self.segment_duration = segment_duration
        self.overlap = overlap
        self. sample_rate = sample_rate
        self.device = device
        self.crackle_threshold = crackle_threshold
        self.wheeze_threshold = wheeze_threshold
        
        # Load model
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
        
        print(f"\n⚙️  Configuration:")
        print(f"   Segment duration: {segment_duration}s")
        print(f"   Overlap: {overlap*100:.0f}%")
        print(f"   Crackle threshold: {crackle_threshold:.2f}")
        print(f"   Wheeze threshold: {wheeze_threshold:.2f}")
    
    def _load_model(self):
        """Load trained model."""
        print(f"Loading model from {self.model_path}...")
        
        checkpoint = torch.load(self.model_path, map_location='cpu', weights_only=False)
        config = checkpoint['config']
        
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
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        print(f"✓ Model loaded")
        return model, config
    
    def load_audio(self, audio_path: str, max_duration: float = 15.0) -> np.ndarray:
        """Load audio file."""
        print(f"\nLoading audio:  {audio_path}")
        audio, sr = librosa.load(audio_path, sr=self.sample_rate, duration=max_duration)
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
        self.model = self.model.to(device)
        self.model.eval()
        
        results = []
        
        # Preprocess
        print("Preprocessing segments...")
        preprocessed = []
        segment_info = []
        
        temp_dir = Path('/tmp/audio_segments')
        temp_dir.mkdir(exist_ok=True)
        
        for idx, (segment, start_time, end_time) in enumerate(tqdm(segments, desc="Preprocessing")):
            temp_path = temp_dir / f'segment_{idx: 04d}.wav'
            sf.write(temp_path, segment, self.sample_rate)
            
            try:
                mel_spec = self.preprocessor.preprocess(str(temp_path))
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
                        predicted_class_idx = predicted_classes[j].item()
                        
                        # Extract confidences
                        normal_conf = float(probs[0])
                        crackle_conf = float(probs[1])
                        wheeze_conf = float(probs[2])
                        both_conf = float(probs[3])
                        
                        # Apply thresholds
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
    
    def visualize_with_spectrogram(
        self,
        results: List[SegmentResult],
        audio:  np.ndarray,
        save_path: str = None,
        show: bool = True
    ):
        """
        Create visualization with mel-spectrogram. 
        
        Layout:
        1. Waveform
        2. Mel-spectrogram
        3. Detection timeline
        """
        fig = plt.figure(figsize=(20, 12))
        
        # Create grid for subplots
        gs = fig. add_gridspec(3, 1, height_ratios=[1, 2, 1.5], hspace=0.3)
        
        ax_wave = fig.add_subplot(gs[0])
        ax_spec = fig.add_subplot(gs[1])
        ax_timeline = fig.add_subplot(gs[2])
        
        duration = len(audio) / self.sample_rate
        
        # Color scheme
        colors = {
            'normal': '#F5F5F5',      # Light gray
            'wheeze': '#22C55E',      # Green
            'crackle': '#9333EA',     # Purple
            'both': '#EF4444'         # Red
        }
        
        # ===== 1. WAVEFORM =====
        time_axis = np.linspace(0, duration, len(audio))
        ax_wave.plot(time_axis, audio, color='#64748B', linewidth=0.6, alpha=0.8)
        ax_wave.set_ylabel('Amplitude', fontsize=11, fontweight='bold')
        ax_wave.set_xlim([0, duration])
        ax_wave.grid(True, alpha=0.2, linestyle='--')
        ax_wave.set_title('Audio Waveform', fontsize=14, fontweight='bold', pad=10)
        ax_wave.set_xticklabels([])
        
        # ===== 2. MEL-SPECTROGRAM =====
        # Compute mel-spectrogram for the entire audio
        n_fft = 2048
        hop_length = 512
        n_mels = 128
        
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2.0
        )
        
        # Convert to dB
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Display spectrogram
        img = librosa.display.specshow(
            mel_spec_db,
            sr=self.sample_rate,
            hop_length=hop_length,
            x_axis='time',
            y_axis='mel',
            ax=ax_spec,
            cmap='viridis'
        )
        
        # Add colorbar
        cbar = fig.colorbar(img, ax=ax_spec, format='%+2.0f dB')
        cbar.set_label('Intensity (dB)', fontsize=10, fontweight='bold')
        
        # Overlay detection markers on spectrogram
        for result in results:
            if result. has_crackle and result.has_wheeze:
                color = colors['both']
                alpha = 0.3
            elif result.has_crackle:
                color = colors['crackle']
                alpha = 0.3
            elif result.has_wheeze:
                color = colors['wheeze']
                alpha = 0.3
            else:
                continue  # Skip normal segments
            
            # Draw semi-transparent overlay
            ax_spec.axvspan(
                result.start_time,
                result.end_time,
                color=color,
                alpha=alpha,
                zorder=10
            )
        
        ax_spec.set_ylabel('Frequency (Hz)', fontsize=11, fontweight='bold')
        ax_spec.set_title('Mel-Spectrogram with Detection Overlay', 
                         fontsize=14, fontweight='bold', pad=10)
        ax_spec.set_xlim([0, duration])
        
        # ===== 3. DETECTION TIMELINE =====
        ax_timeline.set_xlim([0, duration])
        ax_timeline.set_ylim([0, 1])
        
        # Draw colored blocks
        for result in results: 
            # Determine color
            if result.has_crackle and result.has_wheeze:
                color = colors['both']
            elif result.has_crackle:
                color = colors['crackle']
            elif result.has_wheeze:
                color = colors['wheeze']
            else:
                color = colors['normal']
            
            # Draw rectangle
            width = result.end_time - result.start_time
            rect = Rectangle(
                (result.start_time, 0),
                width,
                1,
                facecolor=color,
                edgecolor='#1E293B',
                linewidth=1.5,
                alpha=0.9
            )
            ax_timeline.add_patch(rect)
        
        # Timeline formatting
        ax_timeline.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
        ax_timeline.set_ylabel('')
        ax_timeline.set_yticks([])
        ax_timeline.set_title('Respiratory Sound Detection Timeline', 
                             fontsize=14, fontweight='bold', pad=15)
        ax_timeline.grid(True, axis='x', alpha=0.3, linestyle='--', linewidth=0.8)
        
        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=colors['normal'], edgecolor='#1E293B', label='Normal', linewidth=1.5),
            Patch(facecolor=colors['wheeze'], edgecolor='#1E293B', label='Wheeze', linewidth=1.5),
            Patch(facecolor=colors['crackle'], edgecolor='#1E293B', label='Crackle', linewidth=1.5),
            Patch(facecolor=colors['both'], edgecolor='#1E293B', label='Both', linewidth=1.5),
        ]
        ax_timeline.legend(
            handles=legend_elements,
            loc='upper right',
            fontsize=11,
            framealpha=0.95,
            edgecolor='#1E293B',
            title='Detection Type',
            title_fontsize=11
        )
        
        # Summary statistics
        total = len(results)
        crackle_count = sum(1 for r in results if r. has_crackle and not r.has_wheeze)
        wheeze_count = sum(1 for r in results if r.has_wheeze and not r.has_crackle)
        both_count = sum(1 for r in results if r.has_crackle and r.has_wheeze)
        normal_count = sum(1 for r in results if not r.has_crackle and not r. has_wheeze)
        
        stats_text = (
            f"Summary:  {total} segments | "
            f"Normal: {normal_count} ({100*normal_count/total:.0f}%) | "
            f"Wheeze: {wheeze_count} ({100*wheeze_count/total:.0f}%) | "
            f"Crackle: {crackle_count} ({100*crackle_count/total:.0f}%) | "
            f"Both: {both_count} ({100*both_count/total:.0f}%)"
        )
        
        ax_timeline.text(
            0.02, 0.98, stats_text,
            transform=ax_timeline.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='#1E293B'),
            family='monospace'
        )
        
        # Overall title
        fig.suptitle('Respiratory Sound Analysis with Mel-Spectrogram', 
                    fontsize=16, fontweight='bold', y=0.995)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"\n✓ Spectrogram visualization saved to:  {save_path}")
        
        if show:
            plt. show()
        else:
            plt.close()
    
    def print_summary(self, results: List[SegmentResult]):
        """Print analysis summary."""
        total = len(results)
        
        normal = sum(1 for r in results if not r.has_crackle and not r.has_wheeze)
        crackle_only = sum(1 for r in results if r.has_crackle and not r.has_wheeze)
        wheeze_only = sum(1 for r in results if r.has_wheeze and not r.has_crackle)
        both = sum(1 for r in results if r.has_crackle and r.has_wheeze)
        
        crackle_results = [r for r in results if r.has_crackle]
        wheeze_results = [r for r in results if r. has_wheeze]
        
        avg_crackle_conf = np.mean([r.crackle_confidence for r in crackle_results]) if crackle_results else 0
        avg_wheeze_conf = np.mean([r. wheeze_confidence for r in wheeze_results]) if wheeze_results else 0
        
        print("\n" + "=" * 70)
        print("RESPIRATORY SOUND ANALYSIS SUMMARY")
        print("=" * 70)
        print(f"Total duration: {results[-1].end_time:.2f}s")
        print(f"Total segments: {total}")
        print("\n" + "-" * 70)
        print("DETECTION BREAKDOWN:")
        print("-" * 70)
        print(f"  Normal:         {normal: 3d} ({100*normal/total: 5.1f}%)")
        print(f"  Wheeze only:   {wheeze_only: 3d} ({100*wheeze_only/total:5.1f}%) - Avg conf: {avg_wheeze_conf:.2f}")
        print(f"  Crackle only:  {crackle_only:3d} ({100*crackle_only/total:5.1f}%) - Avg conf: {avg_crackle_conf:.2f}")
        print(f"  Both:          {both:3d} ({100*both/total:5.1f}%)")
        
        if crackle_results:
            print(f"\n🟣 CRACKLE DETECTIONS ({len(crackle_results)} total):")
            for i, r in enumerate(crackle_results[: 5], 1):
                print(f"   {i}. {r.start_time:.2f}s - {r.end_time:.2f}s")
            if len(crackle_results) > 5:
                print(f"   ... and {len(crackle_results) - 5} more")
        
        if wheeze_results: 
            print(f"\n🟢 WHEEZE DETECTIONS ({len(wheeze_results)} total):")
            for i, r in enumerate(wheeze_results[: 5], 1):
                print(f"   {i}.  {r.start_time:.2f}s - {r.end_time:.2f}s")
            if len(wheeze_results) > 5:
                print(f"   ... and {len(wheeze_results) - 5} more")
        
        print("=" * 70)
    
    def export_results(self, results: List[SegmentResult], output_path: str):
        """Export results to CSV."""
        import csv
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Start (s)', 'End (s)', 'Detection Type',
                'Has Crackle', 'Has Wheeze',
                'Crackle Confidence', 'Wheeze Confidence'
            ])
            
            for r in results:
                if r.has_crackle and r.has_wheeze:
                    detection_type = 'Both'
                elif r.has_crackle:
                    detection_type = 'Crackle'
                elif r.has_wheeze:
                    detection_type = 'Wheeze'
                else:
                    detection_type = 'Normal'
                
                writer.writerow([
                    f"{r.start_time:.3f}",
                    f"{r. end_time:.3f}",
                    detection_type,
                    r.has_crackle,
                    r.has_wheeze,
                    f"{r.crackle_confidence:.4f}",
                    f"{r.wheeze_confidence:.4f}"
                ])
        
        print(f"✓ Results exported to: {output_path}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Respiratory sound analyzer with mel-spectrogram visualization"
    )
    parser.add_argument("--audio", type=str, required=True, help="Path to audio file")
    parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--segment-duration", type=float, default=1.0)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--crackle-threshold", type=float, default=0.3)
    parser.add_argument("--wheeze-threshold", type=float, default=0.3)
    parser.add_argument("--output-dir", type=str, default="analysis_results")
    parser.add_argument("--device", type=str, choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument("--no-display", action='store_true')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Create analyzer
    analyzer = SpectrogramAudioAnalyzer(
        model_path=args.model,
        segment_duration=args.segment_duration,
        overlap=args.overlap,
        device=args.device,
        crackle_threshold=args.crackle_threshold,
        wheeze_threshold=args.wheeze_threshold
    )
    
    # Analyze
    results, audio = analyzer.analyze_audio(args.audio)
    
    # Print summary
    analyzer.print_summary(results)
    
    # Visualize with spectrogram
    audio_name = Path(args.audio).stem
    viz_path = output_dir / f"{audio_name}_spectrogram.png"
    analyzer.visualize_with_spectrogram(
        results,
        audio,
        save_path=str(viz_path),
        show=not args.no_display
    )
    
    # Export
    csv_path = output_dir / f"{audio_name}_detections.csv"
    analyzer. export_results(results, str(csv_path))
    
    print(f"\n✓ All results saved to: {output_dir}")


if __name__ == "__main__":
    main()

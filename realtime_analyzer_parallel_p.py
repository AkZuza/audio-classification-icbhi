"""
Optimized parallel analyzer using GPU batch processing.
Faster than sequential processing. 
"""

import torch
import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from tqdm import tqdm
from typing import List
import warnings
warnings.filterwarnings('ignore')

from src.models. cnn import LightweightCNN
from src. models.resnet import CompactResNet
from src.data.preprocessing import AudioPreprocessor
from realtime_analyzer import SegmentResult, ParallelAudioAnalyzer


class BatchAudioAnalyzer(ParallelAudioAnalyzer):
    """Optimized analyzer using batch processing."""
    
    def process_segments_batch(self, segments: List, batch_size: int = 32) -> List[SegmentResult]:
        """
        Process segments in batches for faster inference.
        
        Args:
            segments: List of (segment, start_time, end_time) tuples
            batch_size:  Batch size for processing
        
        Returns:
            List of SegmentResult objects
        """
        device = torch.device(self.device if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(device)
        self.model.eval()
        
        results = []
        
        # Preprocess all segments
        print("Preprocessing segments...")
        preprocessed = []
        segment_info = []
        
        for segment, start_time, end_time in tqdm(segments, desc="Preprocessing"):
            # Save temporarily
            temp_path = Path(f'/tmp/temp_segment_{len(preprocessed)}.wav')
            sf.write(temp_path, segment, self.sample_rate)
            
            # Preprocess
            mel_spec = self.preprocessor.preprocess(str(temp_path))
            preprocessed.append(mel_spec)
            segment_info.append((start_time, end_time))
            
            # Clean up
            temp_path.unlink()
        
        # Process in batches
        print(f"\nProcessing in batches of {batch_size}...")
        
        with torch.no_grad():
            for i in tqdm(range(0, len(preprocessed), batch_size), desc="Inference"):
                batch_specs = preprocessed[i:i + batch_size]
                batch_info = segment_info[i: i + batch_size]
                
                # Stack into batch
                batch_tensor = torch.stack(batch_specs).to(device)
                
                # Inference
                outputs = self.model(batch_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                predicted_classes = outputs.argmax(dim=1)
                
                # Process results
                for j, (start_time, end_time) in enumerate(batch_info):
                    probs = probabilities[j].cpu().numpy()
                    predicted_class_idx = predicted_classes[j]. item()
                    
                    # Extract confidences
                    normal_conf = probs[0]
                    crackle_conf = probs[1]
                    wheeze_conf = probs[2]
                    both_conf = probs[3]
                    
                    # Detection logic
                    has_crackle = (crackle_conf > 0.5) or (both_conf > 0.5)
                    has_wheeze = (wheeze_conf > 0.5) or (both_conf > 0.5)
                    
                    total_crackle_conf = crackle_conf + both_conf
                    total_wheeze_conf = wheeze_conf + both_conf
                    
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
        
        return results
    
    def analyze_audio(self, audio_path: str) -> tuple: 
        """
        Analyze complete audio file using batch processing.
        
        Args:
            audio_path: Path to audio file
        
        Returns:
            Tuple of (results, audio)
        """
        # Load audio
        audio = self.load_audio(audio_path)
        
        # Segment audio
        segments = self.segment_audio(audio)
        
        # Process segments in batches
        results = self. process_segments_batch(segments, batch_size=32)
        
        print(f"✓ Analysis complete!")
        
        return results, audio


def main():
    """Main function for batch analyzer."""
    parser = argparse. ArgumentParser(
        description="Optimized respiratory sound analyzer with batch processing"
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
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for inference (default: 32)"
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
    
    # Create batch analyzer
    analyzer = BatchAudioAnalyzer(
        model_path=args. model,
        segment_duration=args.segment_duration,
        overlap=args.overlap,
        device=args.device
    )
    
    # Analyze audio
    results, audio = analyzer.analyze_audio(args.audio)
    
    # Print summary
    analyzer.print_summary(results)
    
    # Visualize results
    audio_name = Path(args.audio).stem
    viz_path = output_dir / f"{audio_name}_analysis. png"
    analyzer.visualize_results(
        results,
        audio,
        save_path=str(viz_path),
        show=not args.no_display
    )
    
    # Export results
    csv_path = output_dir / f"{audio_name}_results.csv"
    analyzer.export_results(results, str(csv_path))
    
    print(f"\n✓ All results saved to:  {output_dir}")


if __name__ == "__main__": 
    main()
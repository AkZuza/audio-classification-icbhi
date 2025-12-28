"""
Preprocess ICBHI dataset by segmenting audio files based on annotations. 

This script will:
1. Read each audio file and its annotation
2. Extract segments based on start_time and end_time
3. Label each segment as:  normal, crackle, wheeze, or both
4. Save segments as individual .  wav files with proper labels
"""

import os
import librosa
import soundfile as sf
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
import json


class ICBHISegmenter:
    """Segment ICBHI audio files based on respiratory cycle annotations."""
    
    def __init__(self, input_dir, output_dir, sample_rate=16000, min_duration=0.5):
        """
        Initialize segmenter. 
        
        Args:
            input_dir: Directory containing original ICBHI files
            output_dir: Directory to save segmented files
            sample_rate: Target sample rate
            min_duration:  Minimum segment duration in seconds
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.sample_rate = sample_rate
        self.min_duration = min_duration
        
        # Create output directories
        self.create_output_dirs()
        
        # Statistics
        self.stats = {
            'normal': 0,
            'crackle': 0,
            'wheeze': 0,
            'both': 0,
            'total_files': 0,
            'total_segments': 0,
            'skipped_segments': 0
        }
    
    def create_output_dirs(self):
        """Create output directory structure."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for each class
        for class_name in ['normal', 'crackle', 'wheeze', 'both']:
            (self.output_dir / class_name).mkdir(exist_ok=True)
        
        print(f"Created output directory: {self.output_dir}")
    
    def parse_annotation(self, txt_file):
        """
        Parse annotation file. 
        
        Format:  start_time    end_time    crackles    wheezes
        Returns: List of (start, end, crackle, wheeze) tuples
        """
        annotations = []
        
        try:
            with open(txt_file, 'r') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    
                    if len(parts) >= 4:
                        try:
                            start = float(parts[0])
                            end = float(parts[1])
                            crackle = int(parts[2])
                            wheeze = int(parts[3])
                            
                            annotations.append((start, end, crackle, wheeze))
                        except (ValueError, IndexError) as e:
                            print(f"  Warning: Could not parse line in {txt_file. name}: {line.strip()}")
                            continue
        except Exception as e:
            print(f"  Error reading {txt_file}:  {e}")
        
        return annotations
    
    def get_label(self, crackle, wheeze):
        """
        Determine label based on crackle and wheeze flags.
        
        Args:
            crackle: 1 if crackles present, 0 otherwise
            wheeze: 1 if wheezes present, 0 otherwise
        
        Returns:
            Label string:  'normal', 'crackle', 'wheeze', or 'both'
        """
        if crackle == 1 and wheeze == 1:
            return 'both'
        elif crackle == 1:
            return 'crackle'
        elif wheeze == 1:
            return 'wheeze'
        else:
            return 'normal'
    
    def segment_audio(self, audio_path, txt_path):
        """
        Segment a single audio file based on its annotations.
        
        Args:
            audio_path: Path to audio file
            txt_path: Path to annotation file
        
        Returns:
            Number of segments created
        """
        # Load audio
        try:
            audio, sr = librosa.load(audio_path, sr=self. sample_rate)
        except Exception as e:
            print(f"  Error loading {audio_path. name}: {e}")
            return 0
        
        # Parse annotations
        annotations = self.parse_annotation(txt_path)
        
        if not annotations:
            print(f"  Warning: No valid annotations for {audio_path.name}")
            return 0
        
        # Extract segments
        segments_created = 0
        base_name = audio_path.stem
        
        for idx, (start, end, crackle, wheeze) in enumerate(annotations):
            # Calculate sample indices
            start_sample = int(start * self.sample_rate)
            end_sample = int(end * self.sample_rate)
            
            # Extract segment
            segment = audio[start_sample:end_sample]
            
            # Check duration
            duration = len(segment) / self.sample_rate
            
            if duration < self.min_duration:
                self.stats['skipped_segments'] += 1
                continue
            
            # Get label
            label = self.get_label(crackle, wheeze)
            
            # Create filename
            segment_name = f"{base_name}_seg{idx:03d}_{label}.wav"
            output_path = self.output_dir / label / segment_name
            
            # Save segment
            try:
                sf.write(output_path, segment, self.sample_rate)
                segments_created += 1
                self.stats[label] += 1
                self.stats['total_segments'] += 1
            except Exception as e:
                print(f"  Error saving segment {segment_name}: {e}")
        
        return segments_created
    
    def process_all(self):
        """Process all audio files in the input directory."""
        # Find all audio files
        audio_files = list(self.input_dir.glob("*.wav"))
        
        if not audio_files:
            print(f"No .  wav files found in {self.input_dir}")
            return
        
        print(f"\nFound {len(audio_files)} audio files")
        print(f"Sample rate: {self.sample_rate} Hz")
        print(f"Minimum segment duration: {self.min_duration} seconds")
        print("\nProcessing.. .\n")
        
        # Process each file
        pbar = tqdm(audio_files, desc="Segmenting audio files")
        
        for audio_path in pbar:
            txt_path = audio_path.with_suffix('.txt')
            
            if not txt_path.exists():
                pbar.write(f"Warning: No annotation file for {audio_path.name}")
                continue
            
            segments = self.segment_audio(audio_path, txt_path)
            self.stats['total_files'] += 1
            
            pbar.set_postfix({
                'segments': self.stats['total_segments'],
                'normal': self.stats['normal'],
                'crackle': self. stats['crackle'],
                'wheeze': self.stats['wheeze'],
                'both':  self.stats['both']
            })
        
        # Print summary
        self.print_summary()
        
        # Save statistics
        self.save_stats()
    
    def print_summary(self):
        """Print processing summary."""
        print("\n" + "="*60)
        print("SEGMENTATION COMPLETE")
        print("="*60)
        print(f"Files processed: {self.stats['total_files']}")
        print(f"Total segments created: {self.stats['total_segments']}")
        print(f"Segments skipped (too short): {self.stats['skipped_segments']}")
        print("\nClass distribution:")
        print(f"  Normal:    {self.stats['normal']: 4d} ({100*self.stats['normal']/max(1, self.stats['total_segments']):.1f}%)")
        print(f"  Crackle:  {self.stats['crackle']:4d} ({100*self.stats['crackle']/max(1, self.stats['total_segments']):.1f}%)")
        print(f"  Wheeze:    {self.stats['wheeze']:4d} ({100*self.stats['wheeze']/max(1, self.stats['total_segments']):.1f}%)")
        print(f"  Both:     {self.stats['both']:4d} ({100*self.stats['both']/max(1, self.stats['total_segments']):.1f}%)")
        print("="*60)
        print(f"\nSegmented files saved to: {self.output_dir}")
    
    def save_stats(self):
        """Save statistics to JSON file."""
        stats_file = self.output_dir / "segmentation_stats.json"
        
        with open(stats_file, 'w') as f:
            json.dump(self.stats, f, indent=2)
        
        print(f"Statistics saved to:  {stats_file}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Segment ICBHI audio files based on respiratory cycle annotations"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/ICBHI/audio_and_txt_files",
        help="Input directory containing .  wav and . txt files"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/ICBHI_segmented",
        help="Output directory for segmented files"
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Target sample rate (default: 16000)"
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=0.5,
        help="Minimum segment duration in seconds (default: 0.5)"
    )
    
    args = parser.parse_args()
    
    # Create segmenter
    segmenter = ICBHISegmenter(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        sample_rate=args.sample_rate,
        min_duration=args.min_duration
    )
    
    # Process all files
    segmenter.process_all()


if __name__ == "__main__": 
    main()

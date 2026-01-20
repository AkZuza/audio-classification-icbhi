"""Interactive analyzer with live playback and visualization."""

import pygame
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import sounddevice as sd
from pathlib import Path
import argparse

from realtime_analyzer_parallel import BatchAudioAnalyzer


class InteractiveAudioVisualizer:
    """Interactive visualizer with audio playback."""
    
    def __init__(self, audio_path: str, results: list, audio:  np.ndarray, sample_rate: int):
        self.audio_path = audio_path
        self.results = results
        self.audio = audio
        self.sample_rate = sample_rate
        self.duration = len(audio) / sample_rate
        
        # Pygame setup
        pygame.init()
        self.width = 1600
        self.height = 900
        self.screen = pygame. display.set_mode((self. width, self.height))
        pygame.display.set_caption("Respiratory Sound Analyzer")
        
        # Colors
        self.bg_color = (20, 20, 30)
        self.text_color = (255, 255, 255)
        self.crackle_color = (147, 51, 234)  # Purple
        self.wheeze_color = (34, 197, 94)     # Green
        
        # Font
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)
        
        # Playback state
        self.playing = False
        self.current_time = 0.0
        self.playback_speed = 1.0
        
    def draw_timeline(self):
        """Draw the analysis timeline."""
        # Timeline dimensions
        timeline_x = 50
        timeline_y = 400
        timeline_width = self.width - 100
        timeline_height = 200
        
        # Draw background
        pygame.draw.rect(self.screen, (40, 40, 50), 
                        (timeline_x, timeline_y, timeline_width, timeline_height))
        
        # Draw results
        for result in self.results:
            # Calculate position
            x_start = timeline_x + (result. start_time / self.duration) * timeline_width
            x_end = timeline_x + (result.end_time / self.duration) * timeline_width
            width = x_end - x_start
            
            # Draw crackle
            if result.has_crackle:
                height = result.crackle_confidence * (timeline_height // 2)
                pygame.draw.rect(self.screen, self.crackle_color,
                               (x_start, timeline_y + timeline_height // 2 - height, 
                                max(width, 2), height))
            
            # Draw wheeze
            if result. has_wheeze:
                height = result.wheeze_confidence * (timeline_height // 2)
                pygame.draw.rect(self.screen, self.wheeze_color,
                               (x_start, timeline_y + timeline_height // 2, 
                                max(width, 2), height))
        
        # Draw current time indicator
        current_x = timeline_x + (self. current_time / self.duration) * timeline_width
        pygame.draw.line(self.screen, (255, 255, 0), 
                        (current_x, timeline_y), 
                        (current_x, timeline_y + timeline_height), 3)
        
        # Draw time labels
        for i in range(6):
            t = (i / 5) * self.duration
            x = timeline_x + (t / self.duration) * timeline_width
            label = self.small_font.render(f"{t:.1f}s", True, self.text_color)
            self.screen.blit(label, (x - 20, timeline_y + timeline_height + 10))
    
    def draw_info(self):
        """Draw information panel."""
        # Title
        title = self.font.render("Respiratory Sound Analysis", True, self.text_color)
        self.screen.blit(title, (50, 30))
        
        # File name
        filename = self.small_font.render(f"File: {Path(self.audio_path).name}", 
                                         True, self. text_color)
        self.screen.blit(filename, (50, 80))
        
        # Current time
        time_text = self.font.render(f"Time: {self.current_time:. 2f}s / {self.duration:.2f}s", 
                                    True, self.text_color)
        self.screen.blit(time_text, (50, 650))
        
        # Legend
        crackle_label = self.font.render("■ Crackles", True, self.crackle_color)
        wheeze_label = self.font. render("■ Wheezes", True, self.wheeze_color)
        self.screen.blit(crackle_label, (50, 700))
        self.screen.blit(wheeze_label, (250, 700))
        
        # Controls
        controls = [
            "SPACE: Play/Pause",
            "R:  Restart",
            "ESC: Exit"
        ]
        for i, control in enumerate(controls):
            text = self.small_font.render(control, True, self.text_color)
            self.screen.blit(text, (50, 750 + i * 30))
        
        # Current detections
        current_result = self.get_current_result()
        if current_result:
            y_offset = 120
            
            if current_result.has_crackle:
                crackle_text = self.font.render(
                    f"CRACKLE DETECTED ({current_result.crackle_confidence:.2%})", 
                    True, self.crackle_color
                )
                self.screen. blit(crackle_text, (50, y_offset))
                y_offset += 40
            
            if current_result.has_wheeze:
                wheeze_text = self.font.render(
                    f"WHEEZE DETECTED ({current_result.wheeze_confidence:.2%})", 
                    True, self.wheeze_color
                )
                self.screen. blit(wheeze_text, (50, y_offset))
    
    def get_current_result(self):
        """Get result for current time."""
        for result in self.results:
            if result.start_time <= self. current_time <= result.end_time:
                return result
        return None
    
    def run(self):
        """Run the interactive visualizer."""
        clock = pygame.time.Clock()
        running = True
        
        while running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame. KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        self. playing = not self.playing
                        if self.playing:
                            # Start audio playback
                            start_sample = int(self.current_time * self.sample_rate)
                            sd.play(self.audio[start_sample:], self.sample_rate)
                        else:
                            sd.stop()
                    elif event.key == pygame.K_r:
                        self.current_time = 0.0
                        self.playing = False
                        sd.stop()
                    elif event.key == pygame.K_ESCAPE:
                        running = False
            
            # Update
            if self.playing:
                self.current_time += 1.0 / 60.0  # Assuming 60 FPS
                if self.current_time >= self. duration:
                    self.current_time = 0.0
                    self.playing = False
                    sd.stop()
            
            # Draw
            self.screen.fill(self.bg_color)
            self.draw_timeline()
            self.draw_info()
            
            pygame.display.flip()
            clock.tick(60)
        
        pygame.quit()
        sd.stop()


def main():
    """Main function for interactive analyzer."""
    parser = argparse.ArgumentParser(description="Interactive respiratory sound analyzer")
    parser.add_argument("--audio", type=str, required=True, help="Path to audio file")
    parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--segment-duration", type=float, default=1.0)
    parser.add_argument("--overlap", type=float, default=0.5)
    
    args = parser.parse_args()
    
    # Analyze audio
    print("Analyzing audio...")
    analyzer = BatchAudioAnalyzer(
        model_path=args.model,
        segment_duration=args. segment_duration,
        overlap=args.overlap
    )
    
    results, audio = analyzer.analyze_audio(args.audio)
    analyzer.print_summary(results)
    
    # Launch interactive visualizer
    print("\nLaunching interactive visualizer...")
    print("Controls:")
    print("  SPACE: Play/Pause")
    print("  R: Restart")
    print("  ESC:  Exit")
    
    visualizer = InteractiveAudioVisualizer(
        audio_path=args.audio,
        results=results,
        audio=audio,
        sample_rate=analyzer.sample_rate
    )
    
    visualizer.run()


if __name__ == "__main__": 
    main()

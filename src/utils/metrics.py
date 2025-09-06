from typing import Dict, List, Optional
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import cv2
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class TrainingMetrics:
    """Class to handle training metrics visualization and video recording."""
    
    def __init__(self, save_dir: str = "results"):
        """Initialize the metrics handler.
        
        Args:
            save_dir: Directory to save results
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize metrics storage
        self.rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.value_losses: List[float] = []
        self.policy_losses: List[float] = []
        
        # Initialize video writer
        self.video_writer = None
        self.video_path = None
        
        # Set style for plots
        sns.set_theme(style="darkgrid")
        sns.set_palette("husl")
        
    def add_metrics(self, info_dict: Dict) -> None:
        """Add metrics from training info dictionary.
        
        Args:
            info_dict: Dictionary containing training metrics
        """
        if 'episode' in info_dict:
            self.rewards.append(info_dict['episode']['r'])
            self.episode_lengths.append(info_dict['episode']['l'])
        if 'value_loss' in info_dict:
            self.value_losses.append(info_dict['value_loss'])
        if 'policy_loss' in info_dict:
            self.policy_losses.append(info_dict['policy_loss'])
            
    def plot_metrics(self) -> None:
        """Generate and save plots for all tracked metrics."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training Metrics', fontsize=16, y=1.02)
        
        # Plot rewards
        self._plot_metric(
            axes[0, 0], 
            self.rewards, 
            'Episode Rewards', 
            'Episode', 
            'Reward',
            window=10
        )
        
        # Plot episode lengths
        self._plot_metric(
            axes[0, 1], 
            self.episode_lengths, 
            'Episode Lengths', 
            'Episode', 
            'Steps',
            window=10
        )
        
        # Plot value loss
        self._plot_metric(
            axes[1, 0], 
            self.value_losses, 
            'Value Loss', 
            'Update', 
            'Loss',
            window=5
        )
        
        # Plot policy loss
        self._plot_metric(
            axes[1, 1], 
            self.policy_losses, 
            'Policy Loss', 
            'Update', 
            'Loss',
            window=5
        )
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(self.save_dir / f'training_metrics_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save metrics as JSON
        self._save_metrics(timestamp)
        
    def _plot_metric(
        self, 
        ax: plt.Axes,
        data: List[float],
        title: str,
        xlabel: str,
        ylabel: str,
        window: int = 10
    ) -> None:
        """Plot a single metric with rolling mean and confidence interval.
        
        Args:
            ax: Matplotlib axes to plot on
            data: List of values to plot
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            window: Window size for rolling mean
        """
        if not data:
            return
            
        values = np.array(data)
        indices = np.arange(len(values))
        
        # Calculate rolling mean
        rolling_mean = np.convolve(values, np.ones(window)/window, mode='valid')
        
        # Plot raw data with low alpha
        ax.plot(indices, values, alpha=0.3, color='gray', label='Raw')
        
        # Plot rolling mean
        ax.plot(indices[window-1:], rolling_mean, linewidth=2, label=f'Mean (window={window})')
        
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
    def _save_metrics(self, timestamp: str) -> None:
        """Save metrics to JSON file.
        
        Args:
            timestamp: Timestamp string for filename
        """
        metrics_dict = {
            'rewards': self.rewards,
            'episode_lengths': self.episode_lengths,
            'value_losses': self.value_losses,
            'policy_losses': self.policy_losses
        }
        
        with open(self.save_dir / f'metrics_{timestamp}.json', 'w') as f:
            json.dump(metrics_dict, f)
            
    def start_video_recording(self, fps: int = 30) -> None:
        """Start recording video of the simulation.
        
        Args:
            fps: Frames per second for the output video
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.video_path = str(self.save_dir / f'simulation_{timestamp}.mp4')
        self.video_writer = cv2.VideoWriter(
            self.video_path,
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps,
            (84, 84)  # Assuming this is the image size from the environment
        )
        logger.info(f"Started video recording: {self.video_path}")
        
    def add_video_frame(self, frame: np.ndarray) -> None:
        """Add a frame to the video recording.
        
        Args:
            frame: RGB frame from the environment
        """
        if self.video_writer is not None:
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            self.video_writer.write(frame_bgr)
            
    def stop_video_recording(self) -> None:
        """Stop video recording and release resources."""
        if self.video_writer is not None:
            self.video_writer.release()
            logger.info(f"Saved video recording to: {self.video_path}")
            self.video_writer = None
            self.video_path = None

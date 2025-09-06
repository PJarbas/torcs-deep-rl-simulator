"""
Script to run a trained agent in the TORCS environment.

This script loads a trained agent and runs it in the TORCS environment,
optionally recording videos of the episodes and collecting performance metrics.

Example:
    python scripts/play_agent.py --model models/ppo_torcs --episodes 5 --record-video

The script will:
1. Load the trained model
2. Run the specified number of episodes
3. Record videos if requested
4. Display real-time metrics
5. Save a summary of the agent's performance
"""

import argparse
import logging
from pathlib import Path
import numpy as np
import cv2
from datetime import datetime
import json
from typing import Dict, List

from envs.torcs_env import TorcsEnv
from agents.ppo_agent import PPOAgent
from utils.metrics import TrainingMetrics

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EpisodeRecorder:
    """Helper class to record episodes and collect metrics."""
    
    def __init__(self, output_dir: str = "results"):
        self.metrics = TrainingMetrics(save_dir=output_dir)
        self.episode_stats: List[Dict] = []
        
    def record_episode(self, episode_num: int, total_reward: float, total_steps: int) -> None:
        """Record episode statistics.
        
        Args:
            episode_num: Episode number
            total_reward: Total reward for the episode
            total_steps: Total steps in the episode
        """
        stats = {
            "episode": episode_num,
            "reward": float(total_reward),
            "steps": total_steps,
            "timestamp": datetime.now().isoformat()
        }
        self.episode_stats.append(stats)
        
    def save_summary(self, output_path: Path) -> None:
        """Save episode statistics summary.
        
        Args:
            output_path: Path to save the summary
        """
        rewards = [s["reward"] for s in self.episode_stats]
        steps = [s["steps"] for s in self.episode_stats]
        
        summary = {
            "episodes": len(self.episode_stats),
            "total_steps": sum(steps),
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "mean_steps": float(np.mean(steps)),
            "std_steps": float(np.std(steps)),
            "min_reward": float(np.min(rewards)),
            "max_reward": float(np.max(rewards)),
            "episode_stats": self.episode_stats
        }
        
        with open(output_path / "evaluation_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
            
        logger.info(f"Saved evaluation summary to {output_path/'evaluation_summary.json'}")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Path to the trained model")
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes to run")
    parser.add_argument("--render", type=bool, default=True, help="Whether to render the environment")
    parser.add_argument("--record-video", action="store_true", help="Record videos of the episodes")
    parser.add_argument("--output", default="results/evaluation", help="Output directory for videos and metrics")
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create environment
    env = TorcsEnv(vision=True, rendering=args.render)
    
    # Create and load agent
    agent = PPOAgent(env)
    agent.load(args.model)
    
    # Initialize recorder
    recorder = EpisodeRecorder(str(output_dir))
    
    try:
        for episode in range(args.episodes):
            logger.info(f"\nStarting episode {episode + 1}/{args.episodes}")
            
            if args.record_video:
                recorder.metrics.start_video_recording()
                
            obs = env.reset()
            done = False
            total_reward = 0
            step = 0
            
            while not done:
                # Get action from agent
                action = agent.act(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                
                # Update metrics
                total_reward += reward
                step += 1
                
                # Record video frame if enabled
                if args.record_video:
                    frame = env.render(mode='rgb_array')
                    recorder.metrics.add_video_frame(frame)
                
                # Display progress
                if step % 10 == 0:
                    logger.info(f"Step: {step}, Reward: {total_reward:.2f}")
            
            # Episode complete
            logger.info(f"\nEpisode {episode + 1} complete - Steps: {step}, Reward: {total_reward:.2f}")
            
            # Stop video recording
            if args.record_video:
                recorder.metrics.stop_video_recording()
            
            # Record episode stats
            recorder.record_episode(episode + 1, total_reward, step)
            
        # Save final summary
        recorder.save_summary(output_dir)
        
    except KeyboardInterrupt:
        logger.info("\nEvaluation interrupted by user")
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        raise
    finally:
        env.close()
        if args.record_video:
            recorder.metrics.stop_video_recording()

if __name__ == "__main__":
    main()

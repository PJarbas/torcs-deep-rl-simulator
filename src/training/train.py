import argparse
import yaml
import logging
from pathlib import Path
from src.envs.torcs_env import TorcsEnv
from src.utils.metrics import TrainingMetrics
from stable_baselines3.common.callbacks import BaseCallback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MetricsCallback(BaseCallback):
    """Callback to track metrics during training."""
    
    def __init__(self, metrics: TrainingMetrics):
        super().__init__()
        self.metrics = metrics
        self.recording = False
        
    def _on_step(self) -> bool:
        # Add metrics from the last step
        self.metrics.add_metrics(self.locals.get('infos', [{}])[0])
        
        # Record video periodically
        if self.n_calls % 1000 == 0:  # Every 1000 steps
            if not self.recording:
                self.metrics.start_video_recording()
                self.recording = True
        elif self.recording and self.n_calls % 1000 == 100:  # Record for 100 steps
            self.metrics.stop_video_recording()
            self.recording = False
            
        # Add frame to video if recording
        if self.recording:
            obs = self.training_env.envs[0].render()
            self.metrics.add_video_frame(obs)
            
        return True
        
    def _on_rollout_end(self) -> None:
        # Plot metrics every rollout
        self.metrics.plot_metrics()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default="results", help="Output directory for results")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize environment
    env = TorcsEnv(vision=config.get("vision", True))

    # Initialize metrics
    metrics = TrainingMetrics(save_dir=str(output_dir))
    callback = MetricsCallback(metrics)

    # Initialize agent
    algo = config.get("algorithm", "PPO")
    if algo == "PPO":
        from src.agents.ppo_agent import PPOAgent
        agent = PPOAgent(env, **config.get("agent_kwargs", {}))
    elif algo == "SAC":
        from src.agents.sac_agent import SACAgent
        agent = SACAgent(env, **config.get("agent_kwargs", {}))
    else:
        raise ValueError(f"Unknown algorithm: {algo}")

    # Train agent
    logger.info("Starting training...")
    try:
        agent.train(
            total_timesteps=config.get("total_timesteps", 1_000_000),
            callback=callback
        )
        logger.info("Training completed successfully")
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise
    finally:
        # Save final metrics plot
        metrics.plot_metrics()
        
        # Save trained model
        model_path = output_dir / "final_model"
        agent.save(str(model_path))
        logger.info(f"Saved final model to {model_path}")
        
if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()

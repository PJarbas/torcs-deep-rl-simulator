import argparse
import yaml
import logging
from pathlib import Path
from src.envs.torcs_env import TorcsEnv
from src.utils.metrics import TrainingMetrics
from stable_baselines3.common.callbacks import BaseCallback

# Configure logging
def setup_logging(output_dir: Path) -> None:
    """Configure logging to both file and console."""
    log_file = output_dir / "training.log"
    
    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(file_formatter)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

# Get module logger
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
    parser.add_argument("--config", required=True, help="Path to training configuration file")
    parser.add_argument("--output", default="results/training", help="Output directory for results")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    setup_logging(output_dir)
    if args.debug:
        logger.setLevel(logging.DEBUG)

    # Load and validate config
    try:
        with open(args.config) as f:
            config = yaml.safe_load(f)
        logger.debug(f"Loaded configuration: {config}")
    except Exception as e:
        logger.error(f"Failed to load config file: {e}")
        raise

    # Get environment config
    env_config = config.get("env_config", {})
    logger.info("Initializing TORCS environment with config:")
    for key, value in env_config.items():
        logger.info(f"  {key}: {value}")

    # Initialize environment
    try:
        env = TorcsEnv(**env_config)
        logger.info("TORCS environment initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize TORCS environment: {e}")
        raise

    # Initialize metrics and callback
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

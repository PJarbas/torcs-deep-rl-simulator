from typing import Any, Dict, Optional, Union

import numpy as np
import numpy.typing as npt
import torch
from gymnasium import spaces
from gymnasium.core import Env
from stable_baselines3 import PPO

from .base_agent import BaseAgent
from .features_extractor import CnnFeatureExtractor


class PPOAgent(BaseAgent):
    def __init__(self, env: Env, **kwargs: Any) -> None:
        """Initialize the PPO agent.

        Args:
            env: The environment to train/run on
            **kwargs: Additional arguments to pass to PPO
        """
        # Determine the device to use
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7:
            device = "cuda"
            print("Using CUDA device:", torch.cuda.get_device_name())
        else:
            device = "cpu"
            print("CUDA device not available or not compatible. Using CPU.")

        # Select the appropriate policy and setup policy kwargs
        if isinstance(env.observation_space, spaces.Dict):
            policy = "MultiInputPolicy"

            # Ensure policy_kwargs exists
            if "policy_kwargs" not in kwargs:
                kwargs["policy_kwargs"] = {}

            # Set the feature extractor class
            kwargs["policy_kwargs"]["features_extractor_class"] = CnnFeatureExtractor

            # Set default features_extractor_kwargs if not provided
            if "features_extractor_kwargs" not in kwargs["policy_kwargs"]:
                kwargs["policy_kwargs"]["features_extractor_kwargs"] = {
                    "features_dim": 512
                }

        elif isinstance(env.observation_space, spaces.Box):
            if len(env.observation_space.shape) == 3:  # Image observation
                policy = "CnnPolicy"
            else:  # Vector observation
                policy = "MlpPolicy"
        else:
            raise ValueError(f"Unsupported observation space: {env.observation_space}")

        # Add device parameter to kwargs
        kwargs["device"] = device

        # Use verbose=1 to see training progress
        kwargs.setdefault("verbose", 1)

        print("Policy kwargs:", kwargs.get("policy_kwargs", {}))
        self.model = PPO(policy, env, **kwargs)

    def act(
        self,
        observation: Dict[str, Union[npt.NDArray, Any]],
        deterministic: bool = True,
    ) -> npt.NDArray:
        """Make a decision based on the current observation.

        Args:
            observation: Current observation from the environment
            deterministic: Whether to use deterministic actions

        Returns:
            Action to take in the environment
        """
        action, _ = self.model.predict(observation, deterministic=deterministic)
        return action

    def train(self, total_timesteps: int) -> None:
        """Train the agent.

        Args:
            total_timesteps: Total number of timesteps to train for
        """
        self.model.learn(total_timesteps=total_timesteps)
        # Save the trained model
        self.save("models/ppo_torcs")

    def save(self, path: str) -> None:
        """Save the agent to disk.

        Args:
            path: Path where to save the agent
        """
        self.model.save(path)

    def load(self, path: str) -> None:
        """Load the agent from disk.

        Args:
            path: Path from where to load the agent
        """
        self.model = PPO.load(path)

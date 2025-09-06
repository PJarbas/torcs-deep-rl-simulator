from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union

import numpy as np
import numpy.typing as npt


class BaseAgent(ABC):
    @abstractmethod
    def act(
        self,
        observation: Dict[str, Union[npt.NDArray, Any]],
        deterministic: bool = False,
    ) -> npt.NDArray:
        """Make a decision based on the current observation.

        Args:
            observation: Current observation from the environment
            deterministic: Whether to use deterministic actions

        Returns:
            Action to take in the environment
        """
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """Save the agent to disk.

        Args:
            path: Path where to save the agent
        """
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """Load the agent from disk.

        Args:
            path: Path from where to load the agent
        """
        pass

    @abstractmethod
    def train(self, total_timesteps: int) -> None:
        """Train the agent.

        Args:
            total_timesteps: Total number of timesteps to train for
        """
        pass

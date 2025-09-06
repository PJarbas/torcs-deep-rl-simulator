from typing import Any, Dict, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
import numpy.typing as npt


class TorcsEnv(gym.Env):
    """TORCS environment wrapper for reinforcement learning."""

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self, host: str = "localhost", port: int = 3001, vision: bool = True
    ) -> None:
        """Initialize the TORCS environment.

        Args:
            host: Hostname of the TORCS server
            port: Port of the TORCS server
            vision: Whether to use vision-based observations
        """
        super().__init__()
        self.host = host
        self.port = port
        self.vision = vision
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )
        if self.vision:
            self.observation_space = gym.spaces.Dict(
                {
                    "image": gym.spaces.Box(0, 255, shape=(84, 84, 3), dtype=np.uint8),
                    "state": gym.spaces.Box(
                        -np.inf, np.inf, shape=(10,), dtype=np.float32
                    ),
                }
            )
        else:
            self.observation_space = gym.spaces.Box(
                -np.inf, np.inf, shape=(10,), dtype=np.float32
            )

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, npt.NDArray], Dict[str, Any]]:
        """Reset the environment to its initial state.

        Args:
            seed: Random seed
            options: Additional options for reset

        Returns:
            Tuple of (observations, info)
        """
        # We need this for the environment to be properly deterministic
        super().reset(seed=seed)

        # Initialize any custom options
        if options is not None:
            pass  # Process any custom options here if needed

        # Reset the environment state
        obs = {
            "image": np.zeros((84, 84, 3), dtype=np.uint8),
            "state": np.zeros(10, dtype=np.float32),
        }

        info: Dict[str, Any] = {}
        return obs, info

    def step(
        self, action: npt.NDArray
    ) -> Tuple[Dict[str, npt.NDArray], float, bool, bool, Dict[str, Any]]:
        """Take a step in the environment.

        Args:
            action: Action to take

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        reward = 0.0
        terminated = False  # True quando o episódio termina naturalmente
        truncated = False  # True quando o episódio é truncado (ex: timeout)
        info: Dict[str, Any] = {}
        obs = {
            "image": np.zeros((84, 84, 3), dtype=np.uint8),
            "state": np.zeros(10, dtype=np.float32),
        }
        return obs, reward, terminated, truncated, info

    def render(self, mode: str = "rgb_array") -> npt.NDArray:
        """Render the environment.

        Args:
            mode: Rendering mode

        Returns:
            RGB array of the rendered frame
        """
        return np.zeros((84, 84, 3), dtype=np.uint8)

    def close(self) -> None:
        """Clean up environment resources."""
        pass

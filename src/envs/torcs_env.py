from typing import Tuple, Dict, Any, Optional, Union
import gymnasium as gym
import numpy as np
import numpy.typing as npt
import logging
import socket
import subprocess
import time
import psutil
import os
from pathlib import Path

class TorcsEnv(gym.Env):
    """TORCS environment wrapper for reinforcement learning."""
    
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self, 
        host: str = "localhost", 
        port: int = 3001, 
        vision: bool = True,
        rendering: bool = True,
        auto_start: bool = True
    ) -> None:
        """Initialize the TORCS environment.
        
        Args:
            host: Hostname of the TORCS server
            port: Port of the TORCS server
            vision: Whether to use vision-based observations
            rendering: Whether to render the TORCS window
            auto_start: Whether to automatically start TORCS if not running
        """
        super().__init__()
        self.host = host
        self.port = port
        self.vision = vision
        self.rendering = rendering
        self.auto_start = auto_start
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Try to connect to TORCS
        if not self._is_torcs_running():
            self.logger.warning("TORCS is not running!")
            if self.auto_start:
                self.logger.info("Attempting to start TORCS...")
                self._start_torcs()
                if not self._is_torcs_running():
                    raise RuntimeError("Failed to start TORCS!")
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(3,),
            dtype=np.float32
        )
        if self.vision:
            self.observation_space = gym.spaces.Dict({
                "image": gym.spaces.Box(
                    0, 255,
                    shape=(84, 84, 3),
                    dtype=np.uint8
                ),
                "state": gym.spaces.Box(
                    -np.inf, np.inf,
                    shape=(10,),
                    dtype=np.float32
                ),
            })
        else:
            self.observation_space = gym.spaces.Box(
                -np.inf, np.inf,
                shape=(10,),
                dtype=np.float32
            )

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
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

        # Check if TORCS is running
        if not self._is_torcs_running():
            self.logger.warning("TORCS not running during reset!")
            if self.auto_start:
                self.logger.info("Attempting to restart TORCS...")
                self._start_torcs()
            else:
                raise RuntimeError("TORCS is not running and auto_start is False")
            
        # Initialize any custom options
        if options is not None:
            pass  # Process any custom options here if needed
            
        # Reset the environment state and get initial observation
        try:
            # TODO: Implement actual TORCS communication here
            # For now, return dummy data but log the attempt
            self.logger.info("Reset environment - Getting initial observation")
            obs = {
                "image": np.zeros((84,84,3), dtype=np.uint8),
                "state": np.zeros(10, dtype=np.float32)
            }
            self.logger.debug(f"Initial observation shape: image={obs['image'].shape}, state={obs['state'].shape}")
            
        except Exception as e:
            self.logger.error(f"Error during reset: {str(e)}")
            raise
        
        info: Dict[str, Any] = {"torcs_connected": self._is_torcs_running()}
        return obs, info

    def step(
        self,
        action: npt.NDArray
    ) -> Tuple[Dict[str, npt.NDArray], float, bool, bool, Dict[str, Any]]:
        """Take a step in the environment.
        
        Args:
            action: Action to take
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Check TORCS connection
        if not self._is_torcs_running():
            self.logger.error("Lost connection to TORCS!")
            return self.reset()[0], 0.0, True, False, {"error": "Lost connection to TORCS"}
            
        try:
            # Log action being taken
            self.logger.debug(f"Taking action: {action}")
            
            # TODO: Implement actual TORCS communication here
            # For now, return dummy data but log the attempt
            reward = 0.0
            terminated = False
            truncated = False
            
            obs = {
                "image": np.zeros((84,84,3), dtype=np.uint8),
                "state": np.zeros(10, dtype=np.float32)
            }
            
            # Basic info for debugging
            info = {
                "torcs_connected": self._is_torcs_running(),
                "action_applied": action.tolist(),
                "step_time": time.time()
            }
            
            self.logger.debug(f"Step result - Reward: {reward:.2f}, Done: {terminated}")
            
        except Exception as e:
            self.logger.error(f"Error during step: {str(e)}")
            return self.reset()[0], 0.0, True, False, {"error": str(e)}
            
        return obs, reward, terminated, truncated, info

    def render(self, mode: str = "rgb_array") -> npt.NDArray:
        """Render the environment.
        
        Args:
            mode: Rendering mode
            
        Returns:
            RGB array of the rendered frame
        """
        return np.zeros((84,84,3), dtype=np.uint8)

    def close(self) -> None:
        """Clean up environment resources."""
        self._stop_torcs()
        
    def _is_torcs_running(self) -> bool:
        """Check if TORCS is running by trying to connect to its port."""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1)
                result = s.connect_ex((self.host, self.port))
                return result == 0
        except:
            return False
            
    def _start_torcs(self) -> None:
        """Start TORCS with appropriate configuration."""
        # Build command with options
        cmd = ["torcs"]
        if not self.rendering:
            cmd.extend(["-nofocus", "-nodamage"])
        if self.vision:
            cmd.extend(["-vision"])
            
        try:
            # Start TORCS process
            self.logger.info(f"Starting TORCS with command: {' '.join(cmd)}")
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Wait for TORCS to initialize
            time.sleep(2.0)
            
            # Check if process is running
            if process.poll() is not None:
                stdout, stderr = process.communicate()
                self.logger.error(f"TORCS failed to start! stdout: {stdout.decode()}, stderr: {stderr.decode()}")
                raise RuntimeError("TORCS failed to start")
                
            self.logger.info("TORCS started successfully")
            
        except Exception as e:
            self.logger.error(f"Error starting TORCS: {str(e)}")
            raise
            
    def _stop_torcs(self) -> None:
        """Stop all TORCS processes."""
        try:
            for proc in psutil.process_iter(['pid', 'name']):
                if proc.info['name'] == 'torcs':
                    self.logger.info(f"Terminating TORCS process {proc.info['pid']}")
                    proc.terminate()
                    proc.wait(timeout=5)
        except Exception as e:
            self.logger.error(f"Error stopping TORCS: {str(e)}")
            # Don't raise the exception, just log it

from typing import Dict, Union, Tuple
import torch as th
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import logging
import cv2
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CnnFeatureExtractor(BaseFeaturesExtractor):
    """CNN feature extractor for Dict observation spaces with images and states."""
    
    def __init__(
        self, 
        observation_space: spaces.Dict,
        features_dim: int = 512
    ) -> None:
        """Initialize the CNN feature extractor.
        
        Args:
            observation_space: The observation space
            features_dim: Number of features to extract
        """
        super().__init__(observation_space, features_dim)
        
        # Validate image dimensions from observation space
        if "image" not in observation_space.spaces:
            raise ValueError("Observation space must contain an 'image' key")
            
        img_space = observation_space.spaces["image"]
        if len(img_space.shape) != 3:
            raise ValueError(f"Image space must be 3D, got shape {img_space.shape}")
            
        # Handle both CHW and HWC formats
        if img_space.shape[0] == 3:  # CHW format
            channels, height, width = img_space.shape
        else:  # HWC format
            height, width, channels = img_space.shape
            
        if not ((height == 84 and width == 84) or (height == 3 and width == 84)):
            raise ValueError(f"Image must be either 84x84x3 (HWC) or 3x84x84 (CHW), got shape {img_space.shape}")
            
        # Convert dimensions to NCHW format for CNN
        if height == 84:  # If in HWC format, rearrange to CHW
            self.input_channels = channels
            self.input_height = height
            self.input_width = width
        else:  # Already in CHW format
            self.input_channels = height  # height is actually channels in this case
            self.input_height = width     # width is actually height in this case
            self.input_width = channels   # channels is actually width in this case
            
        logger.info(f"Input format: channels={self.input_channels}, height={self.input_height}, width={self.input_width}")
            
        # CNN for image processing
        self.cnn = nn.Sequential(
            # First conv layer: input -> 42x42x32
            nn.Conv2d(self.input_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            # Second conv layer: 42x42x32 -> 21x21x64
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            # Third conv layer: 21x21x64 -> 11x11x64
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        # Calculate CNN output size
        with th.no_grad():
            # Create dummy tensor with correct dimensions (NCHW format)
            dummy_input = th.zeros(1, self.input_channels, self.input_height, self.input_width)
            try:
                n_flatten = self.cnn(dummy_input).shape[1]
                logger.info(f"CNN will output {n_flatten} features")
            except Exception as e:
                logger.error(f"Error during CNN initialization: {str(e)}")
                logger.error(f"Dummy input shape: {dummy_input.shape}")
                raise
        
        # Validate state dimensions from observation space
        if "state" not in observation_space.spaces:
            raise ValueError("Observation space must contain a 'state' key")
            
        state_space = observation_space.spaces["state"]
        if len(state_space.shape) != 1:
            raise ValueError(f"State space must be 1D, got shape {state_space.shape}")
            
        state_dim = state_space.shape[0]
        self.state_net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
        )
        
        # Fusion layer to combine CNN and state features
        self.fusion = nn.Sequential(
            nn.Linear(n_flatten + 64, features_dim),
            nn.ReLU(),
        )
        
        # Store expected dimensions for validation (in NCHW format)
        self.expected_channels = self.input_channels
        self.expected_height = self.input_height
        self.expected_width = self.input_width
        self.expected_state_dim = state_dim
    
    def forward(self, observations: Dict[str, th.Tensor]) -> th.Tensor:
        """Extract features from the observations.
        
        Args:
            observations: Dict containing "image" and "state" observations
            
        Returns:
            Tensor of extracted features
            
        Raises:
            ValueError: If input dimensions don't match expected shapes
        """
        # Validate inputs
        if "image" not in observations or "state" not in observations:
            raise ValueError("Observations must contain both 'image' and 'state' keys")
            
        img = observations["image"]
        state = observations["state"]
        
        # Log input dimensions for debugging
        logger.debug(f"Image shape: {img.shape}, Expected channels={self.expected_channels}, height={self.expected_height}, width={self.expected_width}")
        logger.debug(f"State shape: {state.shape}, Expected dim={self.expected_state_dim}")

        try:
            # Process images
            if len(img.shape) == 3:  # Single image
                img = img.unsqueeze(0)  # Add batch dimension
            
            # Log shape before processing
            logger.debug(f"Image tensor shape before processing: {img.shape}")
            
            # Normalize and ensure NCHW format
            image_tensor = img.float().div(255.0)
            
            # Validate dimensions (ignoring batch size)
            if img.shape[1:] != (self.expected_channels, self.expected_height, self.expected_width):
                if img.shape[-1] == self.expected_channels:  # If channels are last
                    image_tensor = image_tensor.permute(0, 3, 1, 2)  # NHWC -> NCHW
                else:
                    raise ValueError(
                        f"Invalid image dimensions. Expected (B, {self.expected_channels}, {self.expected_height}, {self.expected_width}), "
                        f"got {img.shape}"
                    )
            
            logger.debug(f"Image tensor shape after processing: {image_tensor.shape}")
            
            # Process through CNN
            image_features = self.cnn(image_tensor)
            
            # Process states
            state = state.float()
            if len(state.shape) == 1:
                state = state.unsqueeze(0)  # Add batch dimension
            state_features = self.state_net(state)
            
            # Combine features
            combined = th.cat([image_features, state_features], dim=1)
            return self.fusion(combined)
            
        except Exception as e:
            logger.error(f"Error in feature extraction: {str(e)}")
            logger.error(f"Image shape: {img.shape}, State shape: {state.shape}")
            raise

class OpenCVFeatureExtractor(BaseFeaturesExtractor):
    """Feature extractor using OpenCV for image processing."""

    def __init__(
        self,
        observation_space: spaces.Dict,
        features_dim: int = 512
    ) -> None:
        """Initialize the OpenCV feature extractor.

        Args:
            observation_space: The observation space
            features_dim: Number of features to extract
        """
        super().__init__(observation_space, features_dim)

        # Validate image dimensions from observation space
        if "image" not in observation_space.spaces:
            raise ValueError("Observation space must contain an 'image' key")

        img_space = observation_space.spaces["image"]
        if len(img_space.shape) != 3:
            raise ValueError(f"Image space must be 3D (HxWxC), got shape {img_space.shape}")

        height, width, channels = img_space.shape
        if height != 84 or width != 84:
            raise ValueError(f"Image must be 84x84, got {height}x{width}")

        # Validate state dimensions from observation space
        if "state" not in observation_space.spaces:
            raise ValueError("Observation space must contain a 'state' key")

        state_space = observation_space.spaces["state"]
        if len(state_space.shape) != 1:
            raise ValueError(f"State space must be 1D, got shape {state_space.shape}")

        self.state_dim = state_space.shape[0]
        self.features_dim = features_dim

    def forward(self, observations: Dict[str, th.Tensor]) -> th.Tensor:
        """Extract features from the observations.

        Args:
            observations: Dict containing "image" and "state" observations

        Returns:
            Tensor of extracted features
        """
        # Extract image features using OpenCV
        img = observations["image"].cpu().numpy()
        gray_img = cv2.cvtColor(img[0], cv2.COLOR_RGB2GRAY)  # Convert to grayscale
        edges = cv2.Canny(gray_img, 100, 200)  # Edge detection
        edges_flat = edges.flatten()[: self.features_dim]  # Flatten and truncate

        # Extract state features
        state_features = observations["state"].cpu().numpy()

        # Combine features
        combined_features = np.concatenate([edges_flat, state_features], axis=0)
        return th.tensor(combined_features, dtype=th.float32).unsqueeze(0)

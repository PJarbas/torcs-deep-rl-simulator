from typing import Dict, Tuple, Union

import torch as th
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CnnFeatureExtractor(BaseFeaturesExtractor):
    """CNN feature extractor for Dict observation spaces with images and states."""

    def __init__(self, observation_space: spaces.Dict, features_dim: int = 512) -> None:
        """Initialize the CNN feature extractor.

        Args:
            observation_space: The observation space
            features_dim: Number of features to extract
        """
        super().__init__(observation_space, features_dim)

        # CNN para processar imagens
        n_input_channels = observation_space["image"].shape[2]

        self.cnn = nn.Sequential(
            # Primeira camada convolucional
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            # Segunda camada
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            # Terceira camada
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Calcula o tamanho da saída do CNN
        with th.no_grad():
            n_flatten = self.cnn(th.zeros(1, n_input_channels, 84, 84)).shape[1]

        # Rede para processar estados
        state_dim = observation_space["state"].shape[0]
        self.state_net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
        )

        # Camada de fusão
        self.fusion = nn.Sequential(
            nn.Linear(n_flatten + 64, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: Dict[str, th.Tensor]) -> th.Tensor:
        """Extract features from the observations.

        Args:
            observations: Dict containing "image" and "state" observations

        Returns:
            Tensor of extracted features
        """
        # Process images
        images = observations["image"].float() / 255.0
        images = images.permute(0, 3, 1, 2)  # NHWC -> NCHW
        image_features = self.cnn(images)

        # Process states
        states = observations["state"].float()
        state_features = self.state_net(states)

        # Combine features
        combined = th.cat([image_features, state_features], dim=1)
        return self.fusion(combined)

    def forward(self, observations):
        # Processa a imagem
        image_tensor = observations["image"].permute(0, 3, 1, 2)  # NHWC -> NCHW
        image_features = self.cnn(image_tensor)

        # Processa o estado
        state_features = self.state_net(observations["state"])

        # Combina as características
        combined = th.cat([image_features, state_features], dim=1)
        return self.fusion(combined)

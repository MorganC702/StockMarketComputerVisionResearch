import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
import numpy as np 
import gymnasium as gym

class CNNLSTMExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int):
        super().__init__(observation_space, features_dim)

        image_space = observation_space.spaces["images"]  # ✅ Access subspace
        n_timeframes, c, h, w = image_space.shape         # e.g., (5, 3, 128, 128)

        self.cnn = nn.Sequential(
            nn.Conv2d(c, 16, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))  # output: (32, 1, 1)
        )

        self.lstm = nn.LSTM(input_size=32, hidden_size=features_dim, batch_first=True)

        self._n_timeframes = n_timeframes
        self._features_dim = features_dim


    def forward(self, observations):
        x = observations["images"]  # shape: [B, T, C, H, W]
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)  # merge batch + time
        x = self.cnn(x)             # → [B*T, 32, 1, 1]
        x = x.view(B, T, -1)        # → [B, T, 32]
        _, (h_n, _) = self.lstm(x)  # h_n: [1, B, features_dim]
        return h_n.squeeze(0)       # [B, features_dim]

    @property
    def features_dim(self):
        return self._features_dim



class CNNLSTMPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            **kwargs,
            features_extractor_class=CNNLSTMExtractor,
            features_extractor_kwargs=dict(features_dim=256),
        )

import torch
import torch.nn as nn
import torchvision.models as models
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
import gymnasium as gym


class CNNLSTMExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int):
        """
        Args:
            observation_space: Dict with {"images": Box}, shape [B, T, N_tf, C, H, W]
                               B = batch size
                               T = timesteps (history length)
                               N_tf = number of timeframes (e.g., 5)
                               C,H,W = image channels + dimensions
            features_dim: output feature dimension (for policy/value heads)
        """
        super().__init__(observation_space, features_dim)

        # --- Get shape of images from the space ---
        image_space = observation_space.spaces["images"]  # shape: (T, N_tf, C, H, W)
        self._time_steps, self._n_timeframes, c, h, w = image_space.shape

        # --- Shared ResNet18 backbone ---
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        # Freeze all but last block + fc
        for param in resnet.parameters():
            param.requires_grad = False
        for param in resnet.layer4.parameters():
            param.requires_grad = True
        for param in resnet.fc.parameters():
            param.requires_grad = True

        # Replace fc with identity → we want raw features (512-dim from resnet18)
        resnet.fc = nn.Identity()
        self.resnet = resnet


        self._features_dim = features_dim


    def forward(self, observations):
        """
        observations["images"]: shape [B, T, N_tf, C, H, W]
        Returns: [B, features_dim]
        """
        x = observations["images"]  # [B, T, N_tf, C, H, W]
        B, T, N_tf, C, H, W = x.shape

        # Merge batch + time + tf for ResNet
        x = x.view(B * T * N_tf, C, H, W)

        # Encode each image
        feats = self.resnet(x)  # [B*T*N_tf, 512]

        # Reshape back: [B, T, N_tf*512]
        feats = feats.view(B, T, N_tf * 512)

        # Pass through LSTM
        _, (h_n, _) = self.lstm(feats)  # h_n: [num_layers, B, features_dim]

        # Take final layer's hidden state
        out = h_n[-1]  # [B, features_dim]

        return out

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

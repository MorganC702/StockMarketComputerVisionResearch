import torch
import torch.nn as nn
import torchvision.models as models
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
import gymnasium as gym
import numpy as np


class CNNExtractor(BaseFeaturesExtractor):
    """
    Flexible CNN feature extractor for dict observations.
    Works with both old (multi-vector) and new (simplified TP/SL) envs.
    """
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int):
        super().__init__(observation_space, features_dim)

        # --- Extract image shape ---
        image_space = observation_space.spaces["image"]
        c, h, w = image_space.shape

        # --- Dynamically compute non-image feature dim ---
        self.extra_keys = [k for k in observation_space.spaces.keys() if k != "image"]
        other_dim = 0
        for k in self.extra_keys:
            other_dim += int(np.prod(observation_space.spaces[k].shape))

        # --- ResNet backbone ---
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # Freeze all but last block for efficiency
        for param in resnet.parameters():
            param.requires_grad = False
        for param in resnet.layer4.parameters():
            param.requires_grad = True
        for param in resnet.fc.parameters():
            param.requires_grad = True

        # Replace fc with identity
        resnet.fc = nn.Identity()
        self.resnet = resnet

        # Project ResNet output
        self.img_projector = nn.Linear(512, features_dim)

        # Final fusion MLP
        self.final_mlp = nn.Sequential(
            nn.Linear(features_dim + other_dim, features_dim),
            nn.ReLU(),
        )

        self.other_dim = other_dim
        self._features_dim = features_dim

    def forward(self, observations):
        # --- Extract tensors from dict ---
        img = observations["image"]  # [B, 3, H, W]
        img_feats = self.resnet(img)
        img_feats = self.img_projector(img_feats)

        # Gather any non-image features dynamically
        other_feats = []
        for k in self.extra_keys:
            tensor = observations[k].float().view(img.shape[0], -1)
            other_feats.append(tensor)
        if other_feats:
            other_feats = torch.cat(other_feats, dim=1)
            combined = torch.cat([img_feats, other_feats], dim=1)
        else:
            combined = img_feats

        return self.final_mlp(combined)

    @property
    def features_dim(self):
        return self._features_dim


class CNNPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            **kwargs,
            features_extractor_class=CNNExtractor,
            features_extractor_kwargs=dict(features_dim=256),
        )

import torch
import torch.nn as nn
import torchvision.models as models
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
import gymnasium as gym

class CNNExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int):
        super().__init__(observation_space, features_dim)

        # Extract image shape from the dict space
        image_space = observation_space.spaces["image"]
        c, h, w = image_space.shape

        # Non-image features size
        pnl_size = observation_space.spaces["unrealized_pnls"].shape[0]
        action_size = observation_space.spaces["actions"].shape[0]
        close_size = observation_space.spaces["closes"].shape[0]
        pos_size = observation_space.spaces["position"].shape[0]
        self.other_dim = pnl_size + action_size + close_size + pos_size  # 10+10+10+1 = 31

        # Load ResNet18 backbone with ImageNet weights
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # Freeze all layers except last block + fc
        for param in resnet.parameters():
            param.requires_grad = False
        for param in resnet.layer4.parameters():
            param.requires_grad = True
        for param in resnet.fc.parameters():
            param.requires_grad = True

        # Replace final fc layer with identity to extract raw features (512-dim)
        resnet.fc = nn.Identity()
        self.resnet = resnet

        # Project image features
        self.img_projector = nn.Linear(512, features_dim)

        # Combine image features with other features
        self.final_mlp = nn.Sequential(
            nn.Linear(features_dim + self.other_dim, features_dim),
            nn.ReLU()
        )

        self._features_dim = features_dim

    def forward(self, observations):
        # Extract image and other features from dict
        img = observations["image"]  # [B, 3, H, W]
        pnl = observations["unrealized_pnls"]  # [B, 10]
        actions = observations["actions"].float()  # [B, 10]
        closes = observations["closes"]  # [B, 10]
        pos = observations["position"].float()  # [B, 1]

        # Pass image through ResNet
        img_feats = self.resnet(img)          # [B, 512]
        img_feats = self.img_projector(img_feats)  # [B, features_dim]

        # Concatenate non-image features
        other_feats = torch.cat([pnl, actions, closes, pos], dim=1)  # [B, 31]

        # Combine
        combined = torch.cat([img_feats, other_feats], dim=1)  # [B, features_dim+31]
        out = self.final_mlp(combined)  # [B, features_dim]
        return out

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

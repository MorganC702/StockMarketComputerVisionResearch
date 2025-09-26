# policy.py
import torch
import torch.nn as nn
from torchvision import models
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class YOLOResNetExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, yolo_encoder, features_dim=512+3):
        super().__init__(observation_space, features_dim)

        self.encoder = yolo_encoder
        for p in self.encoder.parameters():
            p.requires_grad = False

        # --- figure out channel count dynamically ---
        with torch.no_grad():
            dummy = self.encoder("fake_images/img_0.png")  # one fake path
        in_channels = dummy.shape[1]

        resnet = models.resnet18(weights=None)
        resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.resnet_backbone = nn.Sequential(*list(resnet.children())[:-1])
        for p in self.resnet_backbone.parameters():
            p.requires_grad = False

        self._features_dim = 512 + 3  # ResNet features + tabular (close, balance, position)

    def forward(self, obs):
        img = obs["image"]
        tab = obs["features"]

        with torch.no_grad():
            yolo_feats = self.encoder(img)   # [B, C, H, W]

        img_feats = self.resnet_backbone(yolo_feats)
        img_feats = img_feats.view(img_feats.size(0), -1)

        return torch.cat([img_feats, tab], dim=1)

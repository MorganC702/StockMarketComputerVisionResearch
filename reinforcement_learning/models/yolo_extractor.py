import torch
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.utils import get_device
from models.yolo_encoder import YOLOEncoder


class YOLOCNNExtractor(BaseFeaturesExtractor):
    def __init__(
        self, 
        observation_space, 
        features_dim=512, 
        model_path="./models/runs/yolo_run_2/weights/best.pt"
    ):
        super().__init__(observation_space, features_dim)

        self.device = get_device("auto")
        self.encoder = YOLOEncoder(model_path).to(self.device)

        # Infer output channels (usually 3)
        dummy = torch.zeros(1, 3, 640, 640).to(self.device)
        with torch.no_grad():
            out = self.encoder(dummy)
        self.yolo_channels = out.shape[1]

        self.total_channels = self.yolo_channels * 7  # 7 timeframes
        self.cnn_head = nn.Sequential(
            nn.Conv2d(self.total_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        num_features = observation_space["features"].shape[0]
        self.fc = nn.Linear(64 + num_features, features_dim)

    def forward(self, obs):
        imgs = obs["images"].to(self.device)      # [B, 7, 3, H, W]
        features = obs["features"].to(self.device)

        if imgs.dim() == 4:
            imgs = imgs.unsqueeze(0)

        B, T, C, H, W = imgs.shape
        feats_list = []

        for t in range(T):
            feats_t = self.encoder(imgs[:, t])  # [B, 3, H', W']
            feats_list.append(feats_t)

        stacked = torch.cat(feats_list, dim=1)  # [B, 21, H, W]
        cnn_out = self.cnn_head(stacked).squeeze(-1).squeeze(-1)  # [B, 64]
        out = torch.cat([cnn_out, features], dim=1)  # [B, 64 + 6]
        return self.fc(out)


class CustomYOLOPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            features_extractor_class=YOLOCNNExtractor,
            features_extractor_kwargs=dict(features_dim=512),
            **kwargs
        )

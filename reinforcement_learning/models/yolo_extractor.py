import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from models.yolo_encoder import YOLOEncoder


class YOLOCNNExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=512, model_path="./models/yolov8n.pt"):
        super().__init__(observation_space, features_dim)

        self.encoder = YOLOEncoder(model_path)

        # Test encoder with dummy image of new size
        dummy = torch.zeros(1, 3, 224, 224)  # ✅ UPDATED to 224x224
        with torch.no_grad():
            out = self.encoder(dummy)
        self.yolo_channels = out.shape[1]  # Should be 3

        # 7 images × yolo_channels (usually 3) = 21 input channels to CNN
        self.total_channels = self.yolo_channels * 7  # = 21 if 3 channels/image

        self.cnn_head = nn.Sequential(
            nn.Conv2d(self.total_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),  # Output: [B, 64, 1, 1]
        )

        # ✅ FIXED: 6 features not 5
        self.fc = nn.Linear(64 + 6, features_dim)

    def forward(self, obs):
        imgs = obs["images"]  # [B, 7, 3, 224, 224] or [7, 3, 224, 224]

        if imgs.dim() == 4:  # No batch dim → add one
            imgs = imgs.unsqueeze(0)

        B, T, C, H, W = imgs.shape
        feats_list = []
        for t in range(T):
            feats_t = self.encoder(imgs[:, t])  # [B, 3, H', W']
            feats_list.append(feats_t)

        # Concatenate on channel axis → [B, 21, H, W]
        feats = torch.cat(feats_list, dim=1)

        cnn_out = self.cnn_head(feats).squeeze(-1).squeeze(-1)  # [B, 64]
        out = torch.cat([cnn_out, obs["features"]], dim=1)      # [B, 64 + 6]
        return self.fc(out)

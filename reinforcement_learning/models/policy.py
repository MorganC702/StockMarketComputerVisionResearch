import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from models.yolo_encoder import YOLOEncoder


class YOLOCNNExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=512, model_path="./models/yolov8n.pt"):
        super().__init__(observation_space, features_dim)

        self.encoder = YOLOEncoder(model_path)

        # Probe YOLO output
        dummy = torch.zeros(1, 3, 640, 640)
        with torch.no_grad():
            out = self.encoder(dummy)
        self.yolo_channels = out.shape[1]       # = 560

        # multiply by number of frames (7)
        self.total_channels = self.yolo_channels * 7   # = 3920

        self.cnn_head = nn.Sequential(
            nn.Conv2d(self.total_channels, 512, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.fc = nn.Linear(256 + 3, features_dim)

    def forward(self, obs):
        imgs = obs["images"]  # [B,7,3,H,W] or [7,3,H,W]

        if imgs.dim() == 4:  # [7,3,H,W] → add batch
            imgs = imgs.unsqueeze(0)

        B, T, C, H, W = imgs.shape
        feats_list = []
        for t in range(T):
            feats_t = self.encoder(imgs[:, t])   # [B,560,H’,W’]
            feats_list.append(feats_t)

        # Concatenate all 7 frames across channel axis
        feats = torch.cat(feats_list, dim=1)  # [B,3920,H’,W’]

        cnn_out = self.cnn_head(feats).squeeze(-1).squeeze(-1)  # [B,256]
        out = torch.cat([cnn_out, obs["features"]], dim=1)      # [B,259]
        return self.fc(out)

# Backbone.py
import torch
import torch.nn as nn
import cv2
from ultralytics import YOLO

class MarketPerceptionEncoder(nn.Module):
    def __init__(self, backbone_path="../model/runs/yolo_run_2/weights/best.pt", pos_dim=16, out_channels=64):
        super().__init__()
        # Backbone (keep 0–9 = backbone only)
        m = YOLO(backbone_path).model
        self.backbone = nn.Sequential(*list(m.model[:10]))
        self.backbone.eval()

        self.pos_dim = pos_dim
        self.out_channels = out_channels
        self.pos_emb = nn.Embedding(3 * 24 * 60, pos_dim)

        # Fusion conv (now inside encoder)
        self.fusion_conv = None  

        print(f"[INFO] Encoder initialized: {backbone_path}, pos_dim={pos_dim}, out_channels={out_channels}")

    def forward(self, prev_tensor, curr_tensor, prev_img, curr_img):
        # Extract backbone features
        with torch.no_grad():
            feat_curr = self.backbone(curr_tensor)
            feat_prev = self.backbone(prev_tensor)

        # Motion via optical flow
        prev_gray = cv2.cvtColor(prev_img, cv2.COLOR_RGB2GRAY)
        curr_gray = cv2.cvtColor(curr_img, cv2.COLOR_RGB2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None,
                                            0.5, 3, 15, 3, 5, 1.2, 0)
        flow_resized = cv2.resize(flow, (feat_curr.shape[3], feat_curr.shape[2]))
        motion_tensor = torch.from_numpy(flow_resized).permute(2,0,1).unsqueeze(0).float()

        # Positional encoding (column-wise)
        B, C, H, W = feat_curr.shape
        positions = torch.arange(W, device=feat_curr.device).unsqueeze(0).expand(B, W)
        pos = self.pos_emb(positions)  # [B, W, pos_dim]
        pos = pos.permute(0, 2, 1).unsqueeze(2).expand(B, self.pos_dim, H, W)

        # Fuse [features + motion + pos]
        fused = torch.cat([feat_curr, motion_tensor, pos], dim=1)

        if self.fusion_conv is None:
            self.fusion_conv = nn.Conv2d(fused.shape[1], self.out_channels, kernel_size=1).to(fused.device)
            print(f"[INFO] Fusion conv initialized: in={fused.shape[1]}, out={self.out_channels}")

        fused_out = self.fusion_conv(fused)  # [B, out_channels, H, W]
        return fused_out


class MultiTFEncoder(nn.Module):
    def __init__(self, timeframes, backbone_path="yolov8n.pt", pos_dim=16, out_channels=64):
        super().__init__()
        self.encoders = nn.ModuleDict({
            tf: MarketPerceptionEncoder(backbone_path, pos_dim, out_channels)
            for tf in timeframes
        })

    def forward(self, data_dict):
        outputs = []
        for tf, encoder in self.encoders.items():
            fused = encoder(*data_dict[tf])  # [B, out_channels, H, W]
            outputs.append(fused)
        return torch.cat(outputs, dim=1)  # [B, out_channels*TFs, H, W]

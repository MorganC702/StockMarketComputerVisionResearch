# models/yolo_encoder.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO
from torchvision import transforms
from PIL import Image

class YOLOEncoder(nn.Module):
    def __init__(
        self, 
        weight_path="./best.pt"
    ):
        super().__init__()
        
        self.yolo = YOLO(weight_path).model
        self.features = {}
        
        ############################################################
        # Register hooks for intermediate YOLO layers.
        #
        # - These lines “plant hooks” on layers 15, 18, and 21 in the YOLO model.
        # - A forward hook is a function that runs automatically whenever that
        #   layer executes during a forward pass.
        # - Our hook function (_hook) just saves the output of the layer into
        #   self.features under the given key ("P3", "P4", "P5").
        #
        # Flow:
        #   1. Call self.yolo(x) → YOLO forward starts.
        #   2. When layer 15 finishes, hook saves its output in self.features["P3"].
        #   3. When layer 18 finishes, hook saves its output in self.features["P4"].
        #   4. When layer 21 finishes, hook saves its output in self.features["P5"].
        #   5. After the forward pass, we can retrieve them directly:
        #        p3, p4, p5 = self.features["P3"], self.features["P4"], self.features["P5"]
        #
        # In short: the hook makes intermediate feature maps available
        # without changing YOLO itself.
        ############################################################
        self.yolo.model[15].register_forward_hook(self._hook("P3"))
        self.yolo.model[18].register_forward_hook(self._hook("P4"))
        self.yolo.model[21].register_forward_hook(self._hook("P5"))
        ############################################################
        
        
        self.transform = transforms.Compose([
            transforms.Resize((640, 640)),
            transforms.ToTensor()
        ])

    def _hook(self, name):
        def fn(module, input, output):
            self.features[name] = output
        return fn

    def forward(self, img_path_or_tensor):
        # Handle str path or pre-loaded tensor
        if isinstance(img_path_or_tensor, str):
            img = Image.open(img_path_or_tensor).convert("RGB")
            x = self.transform(img).unsqueeze(0)
        else:
            x = img_path_or_tensor  # assume already [B,3,H,W]

        with torch.no_grad():
            _ = self.yolo(x)

        p3, p4, p5 = self.features["P3"], self.features["P4"], self.features["P5"]

        # Upsample p4, p5 → match p3 resolution
        target_size = p3.shape[-2:]
        p4_up = F.interpolate(p4, size=target_size, mode="bilinear", align_corners=False)
        p5_up = F.interpolate(p5, size=target_size, mode="bilinear", align_corners=False)

        # Stack along channel axis
        stacked = torch.cat([p3, p4_up, p5_up], dim=1)  # [B, C3+C4+C5, H, W]
        return stacked


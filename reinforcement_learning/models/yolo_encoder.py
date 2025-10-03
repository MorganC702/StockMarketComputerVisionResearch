import torch
import torch.nn.functional as F
from ultralytics import YOLO
from torchvision import transforms
from PIL import Image


class YOLOEncoder(torch.nn.Module):
    def __init__(self, model_path="./models/runs/yolo_run_2/weights/best.pt", image_size=(640, 640)):
        super().__init__()
        self.yolo = YOLO(model_path).model
        self.image_size = image_size

        self.features = {}
        self.yolo.model[10].register_forward_hook(lambda _, __, out: self._hook("P3", out))
        self.yolo.model[13].register_forward_hook(lambda _, __, out: self._hook("P4", out))
        self.yolo.model[16].register_forward_hook(lambda _, __, out: self._hook("P5", out))

        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
        ])

    def _hook(self, name, output):
        self.features[name] = output

    def preprocess(self, img_path_or_tensor):
        if isinstance(img_path_or_tensor, str):
            img = Image.open(img_path_or_tensor).convert("RGB")
            return self.transform(img).unsqueeze(0)  # [1,3,H,W]
        elif isinstance(img_path_or_tensor, torch.Tensor):
            if img_path_or_tensor.ndim == 3:
                return img_path_or_tensor.unsqueeze(0)
            return img_path_or_tensor
        else:
            raise ValueError("Input must be path or tensor")

    def forward(self, img):
        """
        img: Tensor of shape [B, 3, H, W] — one RGB image per timeframe
        Returns: [B, 3, H, W] — averaged and aligned feature maps from P3, P4, P5
        """
        if img.ndim == 3:
            img = img.unsqueeze(0)  # [1, 3, H, W]

        with torch.no_grad():
            _ = self.yolo(img)

        p3 = self.features["P3"]  # [B, C, H1, W1]
        p4 = self.features["P4"]
        p5 = self.features["P5"]

        # Average over channels → [B, 1, H, W]
        p3_avg = p3.mean(dim=1, keepdim=True)
        p4_avg = p4.mean(dim=1, keepdim=True)
        p5_avg = p5.mean(dim=1, keepdim=True)

        # Upsample to P3's size
        target_size = p3_avg.shape[-2:]
        p4_up = F.interpolate(p4_avg, size=target_size, mode="bilinear", align_corners=False)
        p5_up = F.interpolate(p5_avg, size=target_size, mode="bilinear", align_corners=False)

        # Stack to get [B, 3, H, W]
        features = torch.cat([p3_avg, p4_up, p5_up], dim=1)

        return features


import torch
import torch.nn.functional as F
from ultralytics import YOLO
from torchvision import transforms
from PIL import Image


class YOLOEncoder(torch.nn.Module):
    def __init__(self, model_path="./models/yolov8n.pt", image_size=(640, 640)):
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

    def forward(self, batch_imgs):
        with torch.no_grad():
            _ = self.yolo(batch_imgs)

        p3, p4, p5 = self.features["P3"], self.features["P4"], self.features["P5"]

        target_size = p3.shape[-2:]
        p4_up = F.interpolate(p4, size=target_size, mode="bilinear", align_corners=False)
        p5_up = F.interpolate(p5, size=target_size, mode="bilinear", align_corners=False)

        return torch.cat([p3, p4_up, p5_up], dim=1)  # [B,560,H,W]

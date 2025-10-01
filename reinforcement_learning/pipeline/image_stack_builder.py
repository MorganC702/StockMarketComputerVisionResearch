import torch
import os
from torchvision.io import read_image
from torchvision.transforms.functional import resize

class ImageStackBuilder:
    def __init__(self, timeframes: list[str], image_size=(640, 640)):
        self.timeframes = timeframes
        self.image_size = image_size

    def build_stack(self, image_paths: dict) -> torch.Tensor:
        """
        image_paths: dict of {tf_name: path}
        returns: Tensor [7,3,H,W]
        """
        images = []
        for tf in self.timeframes:
            img_path = image_paths.get(tf)
            if img_path is None or not os.path.exists(img_path):
                print(f"[WARN] Missing image for timeframe {tf}. Using zero tensor.")
                img = torch.zeros((3, *self.image_size), dtype=torch.float32)
            else:
                img = read_image(img_path).float() / 255.0
                img = resize(img, self.image_size)
            images.append(img)

        return torch.stack(images)  # [7,3,H,W]

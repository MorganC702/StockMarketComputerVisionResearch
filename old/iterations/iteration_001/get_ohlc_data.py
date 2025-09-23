"""
File:           get_ohlc_data.py
Description:    Gathers OHLC images and labels, applys transforms if passed in, returns image vectors and labels.
Author:         Morgan Cooper
Created:        2025-09-01
Updated:        2025-09-05

Notes:

"""

from torch.utils.data import Dataset
from PIL import Image

class OHLCImageDataset(Dataset):
    def __init__(
        self,                           
        image_paths,                    # list of image_paths
        labels,                         # list of labels
        transform=None,                 # transform is a pytorch object passed in to preprocess the data    
    ):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

  
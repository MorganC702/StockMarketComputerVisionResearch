from models.yolo_extractor import YOLOCNNExtractor
from gymnasium.spaces import Dict, Box
import numpy as np

class FeatureExtractor:
    def __init__(self, image_shape=(7, 3, 640, 640), feature_dim=5): 

        observation_space = Dict({
            "images": Box(low=0, high=1, shape=image_shape, dtype=np.float32),
            "features": Box(low=-np.inf, high=np.inf, shape=(feature_dim,), dtype=np.float32),
        })

        self.model = YOLOCNNExtractor(observation_space)

    def __call__(self, obs):
        """Forward pass: takes obs dict and returns features."""
        return self.model(obs)

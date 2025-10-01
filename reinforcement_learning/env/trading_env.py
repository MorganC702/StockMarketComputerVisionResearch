import gymnasium as gym
import numpy as np
import torch
from gymnasium.spaces import Dict, Box
from PIL import Image
from torchvision import transforms
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def safe_log_return(p1, p2):
    p1 = max(p1, 1e-8)
    p2 = max(p2, 1e-8)
    return np.log(p2 / p1)


class TradingEnv(gym.Env):
    def __init__(
        self, 
        meta_df, 
        num_numerical_features = 6,
        num_actions = 5,
        image_size=(640, 640)
    ):
        super().__init__()
        
        self.meta_df = meta_df.reset_index(drop=False)
        missing_images = []

        for idx, row in self.meta_df.iterrows():
            for tf in ["1m", "3m", "5m", "15m", "1h", "4h", "1d"]:
                img_path = row.get(f"{tf}_img")
                if not isinstance(img_path, str) or not img_path.strip():
                    missing_images.append((idx, tf, img_path))

        if missing_images:
            msg = "\n".join([f"[Row {i}] Missing {tf}_img: {val}" for (i, tf, val) in missing_images])
            raise ValueError(f"[INIT ERROR] Detected missing image paths in meta_df:\n{msg}")

        self.image_size = image_size
        self.timeframes = ["1m", "3m", "5m", "15m", "1h", "4h", "1d"]

        # Indexing
        self.start_index = 0   # assume first 60 rows are warm-up
        self.current_index = 0

        # Trading state
        self.position = 0
        self.entry_price = None
        self.last_price = None
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        self.steps_since_entry = None
        self.last_action = 0

        # Transforms
        self.transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor()
        ])

        # Observation & action spaces
        self.observation_space = Dict({
            "images": Box(low=0, high=1, shape=(7, 3, *self.image_size), dtype=np.float32),
            "features": Box(low=-np.inf, high=np.inf, shape=(num_numerical_features,), dtype=np.float32),
        })
        
        self.action_space = gym.spaces.Discrete(num_actions)

        logger.info(f"[ENV INIT] Loaded meta with {len(self.meta_df)} rows. Starting at index {self.start_index}")



    def reset(
        self, 
        seed=None, 
        options=None
    ):
        
        logger.debug("[RESET] Called")

        self.current_index = self.start_index
        self.position = 0
        self.entry_price = None
        self.last_price = None
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        self.steps_since_entry = None
        self.last_action = 0

        obs = self._get_observation(reward=0.0)
        return obs, {}


    def step(self, action):
        if self.current_index >= len(self.meta_df):
            logger.warning("[STEP] End of data reached.")
            return self.reset()

        row = self.meta_df.iloc[self.current_index].to_dict()
        close_price = row["close"]

        if self.last_price is None:
            self.last_price = close_price

        reward = 0.0

        # --- Trade logic ---
        if action == 1 and self.position == 0:  # Buy
            self.position = 1
            self.entry_price = close_price
            self.steps_since_entry = 0
            reward = 0.0
        elif action == 2 and self.position == 0:  # Short
            self.position = -1
            self.entry_price = close_price
            self.steps_since_entry = 0
            reward = 0.0
        elif action == 3 and self.position == 1:  # Close Long
            reward = safe_log_return(self.entry_price, close_price)
            self.realized_pnl += reward
            self.position = 0
            self.entry_price = None
            self.steps_since_entry = None
        elif action == 4 and self.position == -1:  # Close Short
            reward = safe_log_return(close_price, self.entry_price)
            self.realized_pnl += reward
            self.position = 0
            self.entry_price = None
            self.steps_since_entry = None
        elif action == 0:
            reward = 0.0
        else:
            reward = -1.0  # Invalid action

        # Update unrealized PnL
        if self.position == 1:
            self.unrealized_pnl = safe_log_return(self.entry_price, close_price)
        elif self.position == -1:
            self.unrealized_pnl = safe_log_return(close_price, self.entry_price)
        else:
            self.unrealized_pnl = 0.0

        if self.position != 0 and self.steps_since_entry is not None:
            self.steps_since_entry += 1

        self.last_price = close_price
        self.last_action = action

        obs = self._get_observation(reward)
        self.current_index += 1
        terminated = self.current_index >= len(self.meta_df)
        truncated = False

        return obs, reward, terminated, truncated, {}

    def _get_observation(self, reward):
        row = self.meta_df.iloc[self.current_index].to_dict()
        images = []

        for tf in self.timeframes:
            img_path = row.get(f"{tf}_img")
            if not img_path or not isinstance(img_path, str):
                logger.warning(f"[OBS] Missing image path for {tf} at index {self.current_index}")
                img_tensor = torch.zeros((3, *self.image_size))  # fallback to blank image
            else:
                img = Image.open(img_path).convert("RGB")
                img_tensor = self.transform(img)
            images.append(img_tensor)

        stacked = torch.stack(images, dim=0)  # [7, 3, H, W]

        features = np.array([
            self.realized_pnl,
            self.unrealized_pnl,
            self.position,
            self.last_action,
            self.steps_since_entry or 0,
            reward
        ], dtype=np.float32)

        obs = {
            "images": stacked.numpy(),
            "features": features
        }

        return obs

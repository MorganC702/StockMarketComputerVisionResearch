from pathlib import Path
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
import csv

class TradingEnv(gym.Env):
    def __init__(self, meta_df: pd.DataFrame, image_size=(640, 640), base_dir="./data/dataset"):
        super().__init__()
        self.meta_df = meta_df.sort_values("timestamp").reset_index(drop=True)
        self.base_dir = Path(base_dir)

        # --- Actions ---
        self.action_space = spaces.Discrete(3)  # 0=hold, 1=buy, 2=sell

        # --- Observation space ---
        c, h, w = 3, image_size[0], image_size[1]
        self.observation_space = spaces.Dict({
            "images": spaces.Box(low=0, high=1, shape=(7, c, h, w), dtype=np.float32),
            "features": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
        })

        # --- State ---
        self.balance = 10000.0
        self.position = 0
        self.current_step = 0

        # --- Preprocessing ---
        self.img_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor()
        ])

        self.timeframes = ["1m", "3m", "5m", "15m", "1h", "4h", "1d"]

    def _resolve_path(self, path_str: str) -> Path:
        """Resolve CSV paths like './dataset/1m/images/frame_00001.png'
        to the real file under self.base_dir."""
        p = Path(path_str)
        if not p.is_absolute():
            try:
                p = self.base_dir / p.relative_to("./dataset")
            except ValueError:
                p = self.base_dir / p
        return p

    def _get_obs(self):
        row = self.meta_df.iloc[self.current_step]

        imgs = []
        for tf in self.timeframes:
            img_path = self._resolve_path(row[tf])
            img = Image.open(img_path).convert("RGB")
            imgs.append(self.img_transform(img).numpy())  # [3,H,W]

        imgs = np.stack(imgs, axis=0)  # [7,3,H,W]

        obs_features = np.array(
            [row["close"], self.balance, self.position],
            dtype=np.float32
        )

        return {"images": imgs, "features": obs_features}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.balance = 10000.0
        self.position = 0
        self.current_step = 0
        return self._get_obs(), {}
    

    def step(self, action):
        row = self.meta_df.iloc[self.current_step]
        prev_close = self.meta_df.loc[self.current_step - 1, "close"] if self.current_step > 0 else row["close"]
        curr_close = row["close"]

        # --- Determine reward (win/loss) ---
        raw_pnl = 0.0
        if self.position == 1:   # long
            raw_pnl = curr_close - prev_close
        elif self.position == -1:  # short
            raw_pnl = prev_close - curr_close

        # win/loss reward
        if raw_pnl > 0:
            reward = 1.0
        elif raw_pnl < 0:
            reward = -1.0
        else:
            reward = 0.0   # optional

        # --- Transaction handling (no balance scaling now) ---
        new_position = self.position
        if action == 1:  # buy
            new_position = 1
        elif action == 2:  # sell
            new_position = -1
        elif action == 0:  # hold
            new_position = self.position

        # --- State update ---
        self.position = new_position
        self.current_step += 1

        terminated = self.current_step >= len(self.meta_df)
        truncated = False

        obs = self._get_obs()
        info = {
            "timestamp": row["timestamp"],
            "step": self.current_step,
            "reward": reward,
            "position": self.position,
        }

        # --- log to CSV ---
        # with open("trade_log.csv", "a") as f:
        #     f.write(f"{row['timestamp']},{self.current_step},{reward},{self.position}\n")

        return obs, reward, terminated, truncated, info

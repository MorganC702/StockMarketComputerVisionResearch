# envs/trading_env.py

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms


class TradingEnv(gym.Env):
    def __init__(self, meta_df: pd.DataFrame, image_size=(640, 640)):
        super().__init__()
        self.meta_df = meta_df.sort_values("timestamp").reset_index(drop=True)

        # --- Actions ---
        self.action_space = spaces.Discrete(3)  # 0=hold, 1=buy, 2=sell

        # --- Observation space ---
        c, h, w = 3, image_size[0], image_size[1]
        self.observation_space = spaces.Dict({
            "image": spaces.Box(low=0, high=1, shape=(c, h, w), dtype=np.float32),
            "features": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)
        })

        # --- Trading state ---
        self.balance = 10000.0
        self.position = 0  # -1 short, 0 flat, +1 long
        self.current_step = 0

        # --- Image preprocessing ---
        self.img_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor()
        ])

    def _get_obs(self):
        # If we've run out of data, return dummy obs
        if self.current_step >= len(self.meta_df):
            return {
                "image": np.zeros((3, 640, 640), dtype=np.float32),
                "features": np.zeros(3, dtype=np.float32)
            }

        row = self.meta_df.iloc[self.current_step]

        # Load + preprocess image
        img = Image.open(row["path"]).convert("RGB")
        img_tensor = self.img_transform(img).numpy()

        # Tabular features
        obs_features = np.array([
            row["close"],
            self.balance,
            self.position
        ], dtype=np.float32)

        return {"image": img_tensor, "features": obs_features}

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

        # --- Base reward from position ---
        reward = 0.0
        if self.position == 1:   # long
            reward = curr_close - prev_close
        elif self.position == -1:  # short
            reward = prev_close - curr_close

        # --- Transaction cost ---
        fee_rate = 0.001  # 0.1% per trade
        transaction_cost = 0.0

        new_position = self.position
        if action == 1:    # buy
            if self.position != 1:  # only pay if changing position
                transaction_cost = curr_close * fee_rate
            new_position = 1
        elif action == 2:  # sell
            if self.position != -1:
                transaction_cost = curr_close * fee_rate
            new_position = -1
        elif action == 0:  # hold
            new_position = self.position

        # --- Over-trading penalty ---
        overtrade_penalty = -0.05 if new_position != self.position else 0.0

        # --- Holding penalty ---
        holding_penalty = -0.01 if new_position != 0 else 0.0

        # --- Final reward ---
        reward = reward - transaction_cost + overtrade_penalty + holding_penalty

        # Update state
        self.position = new_position
        self.balance += reward
        self.current_step += 1
        done = self.current_step >= len(self.meta_df)

        obs = self._get_obs()
        info = {
            "timestamp": row["timestamp"],
            "balance": self.balance,
            "position": self.position,
            "reward_breakdown": {
                "pnl": float(curr_close - prev_close if self.position != 0 else 0.0),
                "transaction_cost": float(-transaction_cost),
                "overtrade_penalty": float(overtrade_penalty),
                "holding_penalty": float(holding_penalty)
            }
        }

        return obs, reward, done, False, info

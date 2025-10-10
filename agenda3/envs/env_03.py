import gymnasium as gym
from gymnasium.spaces import Dict, Box
from pathlib import Path
from PIL import Image
from torchvision import transforms
import numpy as np
import torch
import logging
import csv
from collections import deque

from reward_functions.reward_v11 import compute_reward

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TradingEnv(gym.Env):
    def __init__(
        self,
        meta_df,
        image_size=(128, 128),
        starting_balance=100_000,
        tp_pct=0.003,          # +2% take profit
        sl_pct=-0.001,         # -1% stop loss
        root_dir=None,
        log_path="trading_log_train.csv",
        log_suffix=None,
    ):
        super().__init__()

        # --- core setup ---
        self.meta_df = meta_df.reset_index(drop=True)
        self.image_size = image_size
        self.root_dir = Path(root_dir) if root_dir else Path(".").resolve()
        self.starting_balance = float(starting_balance)
        self.tp_pct = tp_pct
        self.sl_pct = sl_pct

        # --- spaces ---
        self.transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
        ])
        self.observation_space = Dict({
            "image": Box(low=0, high=1, shape=(3, *self.image_size), dtype=np.float32),
            "position": Box(low=0, high=1, shape=(1,), dtype=np.int8),
            "unrealized_pnl": Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
        })
        self.action_space = gym.spaces.Discrete(2)  # 0 = wait, 1 = enter long

        # --- logging ---
        self.base_log_path = Path(log_path)
        if log_suffix:
            self.log_path = self.base_log_path.with_stem(f"{self.base_log_path.stem}_{log_suffix}")
        else:
            self.log_path = self.base_log_path
        self._init_csv()

        self.reset_state()
        logger.info(f"[ENV INIT] Loaded {len(self.meta_df)} rows.")

    def _init_csv(self):
        self.fieldnames = ["step", "timestamp", "event", "reward", "balance", "entry_price", "price"]
        if not self.log_path.exists():
            with open(self.log_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()

    def reset_state(self):
        self.current_index = 0
        self.balance = self.starting_balance
        self.position = 0  # 0 = flat, 1 = long
        self.entry_price = None
        self.episode_reward = 0.0

    def reset(self, seed=42, options=None):
        self.reset_state()
        return self._get_observation(), {}

    def step(self, action):
        if self.current_index >= len(self.meta_df) - 1:
            obs, _ = self.reset()
            return obs, 0.0, True, False, {}

        price = self.meta_df.iloc[self.current_index]["close"]
        event = "wait"
        reward = 0.0

        # --- If flat, decide to enter ---
        if self.position == 0:
            if action == 1:  # enter long
                self.position = 1
                self.entry_price = price
                event = "enter"
            else:
                event = "wait"

        # --- If in trade, check TP/SL conditions ---
        elif self.position == 1:
            pct_move = (price - self.entry_price) / self.entry_price
            if pct_move >= self.tp_pct:
                # Take profit
                self.balance *= (1 + pct_move)
                event = "tp"
                reward = compute_reward("tp")
                self.position = 0
                self.entry_price = None

            elif pct_move <= self.sl_pct:
                # Stop loss
                self.balance *= (1 + pct_move)
                event = "sl"
                reward = compute_reward("sl")
                self.position = 0
                self.entry_price = None

            else:
                # Still holding
                event = "hold"
                reward = compute_reward("hold")

        # Advance time
        self.current_index += 1
        terminated = self.current_index >= len(self.meta_df) - 1
        truncated = False

        obs = self._get_observation()
        self._log_step(self.meta_df.iloc[self.current_index]["timestamp"], event, reward, price)

        info = {
            "reward": reward,
            "balance": self.balance,
            "position": self.position,
            "event": event,
            "price": price,
        }

        self.episode_reward += reward
        return obs, reward, terminated, truncated, info

    def _get_observation(self):
        row = self.meta_df.iloc[self.current_index]
        img_path = self.root_dir / row["5m_img"]
        if img_path.exists():
            img = Image.open(img_path).convert("RGB")
            img_tensor = self.transform(img)
        else:
            img_tensor = torch.zeros((3, *self.image_size))
        unrealized_pnl = 0.0
        if self.position == 1 and self.entry_price:
            price = row["close"]
            unrealized_pnl = (price - self.entry_price) / self.entry_price
        return {
            "image": img_tensor.numpy(),
            "position": np.array([self.position], dtype=np.int8),
            "unrealized_pnl": np.array([unrealized_pnl], dtype=np.float32),
        }

    def _log_step(self, timestamp, event, reward, price):
        with open(self.log_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow({
                "step": self.current_index,
                "timestamp": str(timestamp),
                "event": event,
                "reward": reward,
                "balance": self.balance,
                "entry_price": self.entry_price if self.entry_price else "",
                "price": price,
            })



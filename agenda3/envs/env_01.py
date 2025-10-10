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

from reward_functions.reward_v10 import compute_reward
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TradingEnv(gym.Env):
    def __init__(
        self,
        meta_df,
        image_size=(128, 128),
        starting_balance=100_000,
        risk_per_trade=1,
        leverage=1,
        fee_rate=0.0001,
        root_dir=None,
        log_path="trading_log_train.csv",  # default to training log
        log_suffix=None,
    ):
        super().__init__()
        self.meta_df = meta_df.reset_index(drop=True)
        self.image_size = image_size
        self.root_dir = Path(root_dir) if root_dir else Path(".").resolve()
        self.fee_rate = fee_rate
        self.risk_per_trade = risk_per_trade
        self.starting_balance = float(starting_balance)
        self.leverage = leverage
        self.window_size = int((60 / 5) * 24 * 6)  # ~1 day of 5-min intervals

        self.transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
        ])

        # Buffers for last 10 steps
        self.pnl_buffer = deque(maxlen=10)
        self.action_buffer = deque(maxlen=10)
        self.close_buffer = deque(maxlen=10)

        # --- Spaces ---
        self.observation_space = Dict({
            "image": Box(low=0, high=1, shape=(3, *self.image_size), dtype=np.float32),
            "unrealized_pnls": Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32),
            "actions": Box(low=0, high=1, shape=(10,), dtype=np.int32),
            "closes": Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32),
            "position": Box(low=-1, high=1, shape=(1,), dtype=np.int8),
        })
        self.action_space = gym.spaces.Discrete(2)

        self.global_step = 0

        # --- Log setup ---
        self.base_log_path = Path(log_path)
        if log_suffix:
            self.log_path = self.base_log_path.with_stem(f"{self.base_log_path.stem}_{log_suffix}")
        else:
            self.log_path = self.base_log_path
        self._init_csv()

        self.episode_start = 0
        self.start_index = 0
        self.reset_state()
        logger.info(f"[ENV INIT] Loaded {len(self.meta_df)} rows.")

    def _init_csv(self):
        self.fieldnames = [
            "step", "timestamp", "reward", "balance", "realized_pnl",
            "unrealized_pnl", "position", "action", "trade_pnl", "trade_fee"
        ]
        if not self.log_path.exists():
            with open(self.log_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()

    def set_log_path(self, path: str):
        self.log_path = Path(path)
        self._init_csv()

    def reset_state(self):
        self.current_index = self.start_index
        self.balance = self.starting_balance
        self.position = 1  # start long
        self.entry_price = self.meta_df.iloc[self.current_index]["close"]
        self.position_size = 0.0
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        self.last_trade_pnl = 0.0
        self.last_trade_fee = 0.0
        self.last_action = 1
        self.episode_reward = 0.0

        # Buffers
        self.pnl_buffer = deque([0.0] * 10, maxlen=10)
        self.action_buffer = deque([1] * 10, maxlen=10)
        self.close_buffer = deque([self.entry_price] * 10, maxlen=10)

        # Open long position
        risk_amount = self.balance * self.risk_per_trade
        self.position_size = (risk_amount * self.leverage) / self.entry_price
        self.open_fee = self.fee_rate * self.entry_price * abs(self.position_size)
        self.unrealized_pnl = -self.open_fee

    def reset(self, seed=42, options=None):
        self.episode_start += self.window_size
        if self.episode_start + self.window_size >= len(self.meta_df):
            self.episode_start = 0
        self.start_index = self.episode_start
        self.reset_state()
        return self._get_observation(), {}

    def step(self, action):
        if self.current_index >= len(self.meta_df) - 1:
            return self.reset()[0], 0.0, True, False, {}

        row = self.meta_df.iloc[self.current_index]
        price = row["close"]
        self.last_trade_pnl = 0.0
        self.last_trade_fee = 0.0

        # === Trading Logic ===
        if action != self.last_action:  # closing position
            direction = 1 if self.position == 1 else -1
            gross_pnl = direction * (price - self.entry_price) * self.position_size
            fee_close = self.fee_rate * price * abs(self.position_size)
            net_pnl = gross_pnl - (self.open_fee + fee_close)
            self.balance += net_pnl
            self.realized_pnl += net_pnl
            self.last_trade_pnl = net_pnl
            self.last_trade_fee = self.open_fee + fee_close

            reward = compute_reward(
                action=action,
                last_action=self.last_action,
                net_pnl=net_pnl,
                unrealized_pnl=self.unrealized_pnl,
                starting_balance=self.starting_balance,
            )

            # open opposite position
            risk_amount = self.balance * self.risk_per_trade
            self.position_size = (risk_amount * self.leverage) / price
            self.open_fee = self.fee_rate * price * abs(self.position_size)
            self.entry_price = price
            self.position *= -1
            self.unrealized_pnl = -self.open_fee

        else:  # holding
            gross = (price - self.entry_price) * self.position_size if self.position == 1 \
                else (self.entry_price - price) * self.position_size
            self.unrealized_pnl = gross - self.open_fee

            reward = compute_reward(
                action=action,
                last_action=self.last_action,
                net_pnl=self.last_trade_pnl,
                unrealized_pnl=self.unrealized_pnl,
                starting_balance=self.starting_balance,
            )

        # === Buffers ===
        self.pnl_buffer.append(self.unrealized_pnl)
        self.action_buffer.append(action)
        self.close_buffer.append(price)

        self.last_action = action
        self.episode_reward += reward
        self.current_index += 1

        terminated = self.current_index >= self.start_index + self.window_size
        truncated = False

        obs = self._get_observation()
        self._log_step(row["timestamp"], reward)

        info = {
            "reward": reward,
            "balance": self.balance,
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": self.unrealized_pnl,
            "position": self.position,
            "action": action,
            "trade_pnl": self.last_trade_pnl,
            "trade_fee": self.last_trade_fee,
        }
        return obs, reward, terminated, truncated, info

    def _get_observation(self):
        row = self.meta_df.iloc[self.current_index]
        # ✅ Choose a time frame column (e.g. 5m_img)
        img_path = self.root_dir / row["5m_img"]
        if img_path.exists():
            img = Image.open(img_path).convert("RGB")
            img_tensor = self.transform(img)
        else:
            img_tensor = torch.zeros((3, *self.image_size))
        return {
            "image": img_tensor.numpy(),
            "unrealized_pnls": np.array(self.pnl_buffer, dtype=np.float32),
            "actions": np.array(self.action_buffer, dtype=np.int32),
            "closes": np.array(self.close_buffer, dtype=np.float32),
            "position": np.array([self.position], dtype=np.int8),
        }

    def _log_step(self, timestamp, reward):
        with open(self.log_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow({
                "step": self.global_step,
                "timestamp": str(timestamp),
                "reward": reward,
                "balance": self.balance,
                "realized_pnl": self.realized_pnl,
                "unrealized_pnl": self.unrealized_pnl,
                "position": self.position,
                "action": self.last_action,
                "trade_pnl": self.last_trade_pnl,
                "trade_fee": self.last_trade_fee,
            })
        self.global_step += 1

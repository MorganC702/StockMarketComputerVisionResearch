import gymnasium as gym
from gymnasium.spaces import Dict, Box
from pathlib import Path
from PIL import Image
from torchvision import transforms
import numpy as np
import torch
import logging
from collections import deque
import csv

# --- Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TradingEnv(gym.Env):
    def __init__(self, meta_df, image_size=(128, 128), num_actions=3,
                 leverage=1, starting_balance=100_000, risk_per_trade=1,
                 root_dir=None, n_steps=8, log_path="trading_log.csv",
                 fee_rate=0.0001):   # 0.01% per side
        super().__init__()
        self.root_dir = Path(root_dir) if root_dir else Path(".").resolve()
        self.meta_df = meta_df.reset_index(drop=True)
        self.timeframes = ["5m", "15m", "1h", "4h", "1d"]
        self.image_size = image_size
        self.n_steps = n_steps
        self.fee_rate = fee_rate

        # --- Episodic chunking ---
        self.window_size = 6 * 24 * 60
        self.episode_start = 0

        # Trading config
        self.starting_balance = float(starting_balance)
        self.leverage = leverage
        self.risk_per_trade = risk_per_trade

        # State
        self.start_index = 0
        self.reset_state()

        # Image transform
        self.transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
        ])

        # History buffers
        self.image_buffer = deque(maxlen=n_steps)
        self.feature_buffer = deque(maxlen=n_steps)

        # Spaces
        self.observation_space = Dict({
            "images": Box(low=0, high=1,
                          shape=(n_steps, len(self.timeframes), 3, *self.image_size),
                          dtype=np.float32),
            "features": Box(low=-np.inf, high=np.inf,
                            shape=(n_steps, 6), dtype=np.float32),
        })
        self.action_space = gym.spaces.Discrete(num_actions)

        # CSV log setup
        # CSV log setup
        self.log_path = Path(log_path)
        self.fieldnames = [
            "step", "timestamp",
            "reward", "balance", "realized_pnl",
            "unrealized_pnl", "position", "action",
            "trade_pnl", "trade_fee"   # ✅ added these
        ]
        if not self.log_path.exists():
            with open(self.log_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()
        self.global_step = 0


        logger.info(f"[ENV INIT] Loaded {len(self.meta_df)} rows with "
                    f"{len(self.timeframes)} TFs, history={n_steps} steps.")

    # --- Reset State ---
    def reset_state(self):
        self.current_index = self.start_index
        self.balance = self.starting_balance
        self.position = 0
        self.entry_price = None
        self.position_size = 0.0
        self.open_fee = 0.0
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        self.steps_since_entry = None
        self.last_action = 0
        self.episode_reward = 0.0

    def reset(self, seed=None, options=None):
        self.episode_start += self.window_size
        if self.episode_start + self.window_size >= len(self.meta_df):
            self.episode_start = 0
        self.start_index = self.episode_start
        self.current_index = self.start_index
        self.reset_state()
        self.image_buffer.clear()
        self.feature_buffer.clear()
        obs = self._get_observation(0.0)
        return obs, {}

    # --- Step ---
    def step(self, action):
        if self.current_index >= len(self.meta_df) - 1:
            return self.reset()[0], 0.0, True, False, {}

        row = self.meta_df.iloc[self.current_index]
        close_price = row["close"]

        reward = 0.0
        self.last_trade_pnl = 0.0
        self.last_trade_fee = 0.0

        # --- Trading logic ---
        if action == 1:  # Long
            if self.position == 0:  # open long
                risk_amount = self.balance * self.risk_per_trade
                self.position_size = (risk_amount * self.leverage) / close_price
                self.open_fee = self.fee_rate * close_price * abs(self.position_size)
                self.unrealized_pnl = -self.open_fee
                self.position = 1
                self.entry_price = close_price

            elif self.position == 1:  # close long
                gross_pnl = (close_price - self.entry_price) * self.position_size
                close_fee = self.fee_rate * close_price * abs(self.position_size)
                net_pnl = gross_pnl - (self.open_fee + close_fee)

                self.balance += net_pnl
                self.realized_pnl += net_pnl
                self.last_trade_pnl = net_pnl
                self.last_trade_fee = self.open_fee + close_fee

                reward = 1.2 * net_pnl

                # reset
                self.position = 0
                self.entry_price = None
                self.position_size = 0.0
                self.unrealized_pnl = 0.0
                self.open_fee = 0.0

        elif action == 2:  # Short
            if self.position == 0:  # open short
                risk_amount = self.balance * self.risk_per_trade
                self.position_size = (risk_amount * self.leverage) / close_price
                self.open_fee = self.fee_rate * close_price * abs(self.position_size)
                self.unrealized_pnl = -self.open_fee
                self.position = -1
                self.entry_price = close_price

            elif self.position == -1:  # close short
                gross_pnl = (self.entry_price - close_price) * self.position_size
                close_fee = self.fee_rate * close_price * abs(self.position_size)
                net_pnl = gross_pnl - (self.open_fee + close_fee)

                self.balance += net_pnl
                self.realized_pnl += net_pnl
                self.last_trade_pnl = net_pnl
                self.last_trade_fee = self.open_fee + close_fee

                reward = 1.2 * net_pnl

                # reset
                self.position = 0
                self.entry_price = None
                self.position_size = 0.0
                self.unrealized_pnl = 0.0
                self.open_fee = 0.0

       # --- Unrealized update & reward if holding ---
        if self.position == 1:  # holding long
            gross = (close_price - self.entry_price) * self.position_size
            self.unrealized_pnl = gross - self.open_fee
            reward = self.unrealized_pnl if self.unrealized_pnl < 0 else 0.0

        elif self.position == -1:  # holding short
            gross = (self.entry_price - close_price) * self.position_size
            self.unrealized_pnl = gross - self.open_fee
            reward = self.unrealized_pnl if self.unrealized_pnl < 0 else 0.0

        else:  # flat
            self.unrealized_pnl = 0.0
            # only apply -1 if we did NOT just close a trade
            if self.last_trade_pnl == 0.0:
                reward = -1.0
            # otherwise keep the exit reward (already set above)


        self.last_action = action
        self.episode_reward += reward

        # bookkeeping
        self.current_index += 1
        terminated = self.current_index >= self.start_index + self.window_size
        truncated = False
        obs = self._get_observation(reward)

        info = {
            "reward": reward,
            "balance": self.balance,
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": self.unrealized_pnl,
            "position": self.position,
            "action": self.last_action,
            "trade_pnl": self.last_trade_pnl,
            "trade_fee": self.last_trade_fee
        }

        # log
        with open(self.log_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow({
                "step": self.global_step,
                "timestamp": str(row["timestamp"]),
                **info
            })
        self.global_step += 1

        return obs, reward, terminated, truncated, info

    def _get_observation(self, reward):
        row = self.meta_df.iloc[self.current_index]
        images = []
        for tf in self.timeframes:
            img_path = row.get(f"{tf}_img")
            if isinstance(img_path, str):
                rel_path = Path(img_path)
                if rel_path.parts[0] == "dataset":
                    rel_path = Path(*rel_path.parts[1:])
                abs_path = self.root_dir / rel_path
                img = Image.open(abs_path).convert("RGB")
                img_tensor = self.transform(img)
            else:
                img_tensor = torch.zeros((3, *self.image_size))
            images.append(img_tensor)

        stacked_imgs = torch.stack(images, dim=0)
        features = np.array([
            self.balance,
            self.realized_pnl,
            self.unrealized_pnl,
            self.position,
            self.last_action,
            reward,
        ], dtype=np.float32)

        self.image_buffer.append(stacked_imgs)
        self.feature_buffer.append(features)

        while len(self.image_buffer) < self.n_steps:
            self.image_buffer.append(torch.zeros_like(stacked_imgs))
            self.feature_buffer.append(np.zeros_like(features))

        images_seq = torch.stack(list(self.image_buffer), dim=0)
        features_seq = np.stack(list(self.feature_buffer), axis=0)

        return {"images": images_seq.numpy(), "features": features_seq}

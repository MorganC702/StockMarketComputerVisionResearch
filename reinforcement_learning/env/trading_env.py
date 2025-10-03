import math
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


class TradingEnv(gym.Env):
    def __init__(self, meta_df, image_size=(640, 640), num_actions=3, leverage=10, starting_balance=100_000, risk_per_trade=0.01):
        super().__init__()

        self.meta_df = meta_df.reset_index(drop=False)
        if "close" not in self.meta_df.columns:
            raise ValueError("meta_df must include 'close' column")

        # Image check
        missing_images = []
        for idx, row in self.meta_df.iterrows():
            for tf in ["1m", "3m", "5m", "15m", "1h", "4h", "1d"]:
                img_path = row.get(f"{tf}_img")
                if not isinstance(img_path, str) or not img_path.strip():
                    missing_images.append((idx, tf, img_path))
        if missing_images:
            msg = "\n".join([f"[Row {i}] Missing {tf}_img: {val}" for (i, tf, val) in missing_images])
            raise ValueError(f"[INIT ERROR] Missing image paths:\n{msg}")

        self.image_size = image_size
        self.timeframes = ["1m", "3m", "5m", "15m", "1h", "4h", "1d"]

        # account + risk
        self.starting_balance = float(starting_balance)
        self.balance = float(starting_balance)
        self.equity = float(starting_balance)
        self.leverage = leverage
        self.risk_per_trade = risk_per_trade

        # State vars
        self.start_index = 0
        self.current_index = 0
        self.position = 0   # -1 short, 0 flat, +1 long
        self.entry_price = None
        self.position_size = 0.0
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        self.steps_since_entry = None
        self.last_action = 0
        self.episode_reward = 0.0

        self.transform = transforms.Compose([transforms.Resize(self.image_size), transforms.ToTensor()])

        # Spaces
        self.observation_space = Dict({
            "images": Box(low=0, high=1, shape=(7, 3, *self.image_size), dtype=np.float32),
            "features": Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32),
        })
        self.action_space = gym.spaces.Discrete(num_actions)

        logger.info(f"[ENV INIT] Loaded {len(self.meta_df)} rows. Balance={self.starting_balance:.2f}, Leverage={self.leverage}x")

    def reset(self, seed=None, options=None):
        self.current_index = self.start_index
        self.balance = self.starting_balance
        self.equity = self.starting_balance
        self.position = 0
        self.entry_price = None
        self.position_size = 0.0
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        self.steps_since_entry = None
        self.last_action = 0
        self.episode_reward = 0.0

        obs = self._get_observation(0.0)
        return obs, {}

    def step(self, action):
        row = self.meta_df.iloc[self.current_index].to_dict()
        close_price = row["close"]

        reward = 0.0

        # --- Trading logic ---
        if action == 1:  # Long toggle
            if self.position == 0:
                self._open_position(1, close_price)
            elif self.position == 1:
                reward += 1.5 * self.unrealized_pnl
                self._close_position(close_price)
            else:
                reward -= 0.001

        elif action == 2:  # Short toggle
            if self.position == 0:
                self._open_position(-1, close_price)
            elif self.position == -1:
                reward += 1.5 * self.unrealized_pnl
                self._close_position(close_price)
            else:
                reward -= 0.001

        # --- Unrealized PnL / Equity ---
        if self.position != 0:
            self.unrealized_pnl = self._calc_unrealized(close_price)
        else:
            self.unrealized_pnl = 0.0

        self.equity = self.balance + self.unrealized_pnl
        reward += self.unrealized_pnl   # reward = unrealized equity

        # --- bookkeeping ---
        if self.position != 0 and self.steps_since_entry is not None:
            self.steps_since_entry += 1

        self.last_action = action
        self.episode_reward += reward
        self.current_index += 1

        terminated = self.current_index >= len(self.meta_df)
        truncated = False
        obs = self._get_observation(reward)

        info = {
            "reward": reward,
            "balance": self.balance,
            "equity": self.equity,
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": self.unrealized_pnl,
            "position": self.position,
            "pos_size": self.position_size,
        }

        if self.current_index % 32 == 0:
            print(f"[LOG {self.current_index}] A={action}, R={reward:.2f}, EpR={self.episode_reward:.2f}, "
                  f"Pos={self.position}, Bal={self.balance:.2f}, Eq={self.equity:.2f}, "
                  f"RPNL={self.realized_pnl:.2f}, UPNL={self.unrealized_pnl:.2f}")

        return obs, reward, terminated, truncated, info

    # --- Helpers ---
    def _open_position(self, direction, price):
        risk_amount = self.balance * self.risk_per_trade
        self.position_size = (risk_amount * self.leverage) / price
        self.position = direction
        self.entry_price = price
        self.steps_since_entry = 0
        self.unrealized_pnl = 0.0

    def _close_position(self, price):
        pnl = self._calc_unrealized(price)
        self.realized_pnl += pnl
        self.balance += pnl
        self.position = 0
        self.entry_price = None
        self.position_size = 0.0
        self.steps_since_entry = None
        self.unrealized_pnl = 0.0

    def _calc_unrealized(self, price):
        if self.position == 1:  # long
            return (price - self.entry_price) * self.position_size
        elif self.position == -1:  # short
            return (self.entry_price - price) * self.position_size
        return 0.0

    def _get_observation(self, reward):
        row = self.meta_df.iloc[self.current_index].to_dict()
        images = []
        for tf in self.timeframes:
            img_path = row.get(f"{tf}_img")
            if not img_path or not isinstance(img_path, str):
                img_tensor = torch.zeros((3, *self.image_size))
            else:
                img = Image.open(img_path).convert("RGB")
                img_tensor = self.transform(img)
            images.append(img_tensor)

        stacked = torch.stack(images, dim=0)  # [7, 3, H, W]
        features = np.array([
            self.balance,
            self.equity,
            self.realized_pnl,
            self.unrealized_pnl,
            self.position,
            self.position_size,
            reward,
        ], dtype=np.float32)

        return {"images": stacked.numpy(), "features": features}

import gymnasium as gym
from gymnasium.spaces import Dict, Box
from pathlib import Path
from PIL import Image
from torchvision import transforms
import numpy as np
import torch
import logging

# --- Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TradingEnv(gym.Env):
    def __init__(self, meta_df, image_size=(128, 128), num_actions=3,
                 leverage=10, starting_balance=100_000, risk_per_trade=0.01,
                 root_dir=None):
        super().__init__()
        self.root_dir = Path(root_dir) if root_dir else Path(".").resolve()
        self.meta_df = meta_df.reset_index(drop=True)
        self.timeframes = ["5m", "15m", "1h", "4h", "1d"]
        self.image_size = image_size
        
        # --- Episodic chunking (6 trading days = 8640 minutes @ 1m resolution)
        self.window_size = 6 * 24 * 60
        self.episode_start = 0


        # Validate image paths
        missing = [
            (idx, tf)
            for idx, row in self.meta_df.iterrows()
            for tf in self.timeframes
            if not isinstance(row.get(f"{tf}_img"), str)
        ]
        if missing:
            raise ValueError("[INIT ERROR] Missing image paths:\n" + "\n".join([f"[Row {i}] {tf}_img" for i, tf in missing]))

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

        # Spaces
        self.observation_space = Dict({
            "images": Box(low=0, high=1, shape=(len(self.timeframes), 3, *self.image_size), dtype=np.float32),
            "features": Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32),
        })
        self.action_space = gym.spaces.Discrete(num_actions)

        logger.info(f"[ENV INIT] Loaded {len(self.meta_df)} rows with {len(self.timeframes)} TFs.")

    def reset_state(self):
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

    def reset(self, seed=None, options=None):
        self.episode_start += self.window_size
        if self.episode_start + self.window_size >= len(self.meta_df):
            self.episode_start = 0  # loop back if we run out

        self.start_index = self.episode_start
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
        if self.current_index >= len(self.meta_df) - 1:
            return self.reset()[0], 0.0, True, False, {}
        
        row = self.meta_df.iloc[self.current_index]
        close_price = row["close"]
        reward = 0.0

        # Trading logic
        if action == 1:  # Long
            if self.position == 0:
                self._open_position(1, close_price)
            elif self.position == 1:
                reward += 1.5 * self.unrealized_pnl
                self._close_position(close_price)
            else:
                reward -= 0.001

        elif action == 2:  # Short
            if self.position == 0:
                self._open_position(-1, close_price)
            elif self.position == -1:
                reward += 1.5 * self.unrealized_pnl
                self._close_position(close_price)
            else:
                reward -= 0.001

        # Unrealized PnL
        self.unrealized_pnl = self._calc_unrealized(close_price) if self.position != 0 else 0.0
        self.equity = self.balance + self.unrealized_pnl
        reward += self.unrealized_pnl
        self.episode_reward += reward
        self.last_action = action
        if self.position != 0 and self.steps_since_entry is not None:
            self.steps_since_entry += 1

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
            print(f"[STEP {self.current_index}] A={action}, R={reward:.2f}, EpR={self.episode_reward:.2f}, "
                  f"Pos={self.position}, Bal={self.balance:.2f}, Eq={self.equity:.2f}, "
                  f"RPNL={self.realized_pnl:.2f}, UPNL={self.unrealized_pnl:.2f}")

        terminated = self.current_index >= self.start_index + self.window_size

        return obs, reward, terminated, truncated, info

    # --- Internal helpers ---
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
        if self.position == 1:
            return (price - self.entry_price) * self.position_size
        elif self.position == -1:
            return (self.entry_price - price) * self.position_size
        return 0.0

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

        stacked_imgs = torch.stack(images, dim=0)  # [T, 3, H, W]
        features = np.array([
            self.balance,
            self.equity,
            self.realized_pnl,
            self.unrealized_pnl,
            self.position,
            self.position_size,
            reward,
        ], dtype=np.float32)

        return {"images": stacked_imgs.numpy(), "features": features}

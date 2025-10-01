import gymnasium as gym
import numpy as np
from gymnasium.spaces import Dict, Box
from pipeline.orchestrator import PipelineOrchestrator
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TradingEnv(gym.Env):
    def __init__(self, meta_df):
        super().__init__()
        self.meta_df = meta_df.reset_index(drop=True)
        self.current_index = 0
        self.start_index = 0

        # Trade state
        self.position = 0
        self.last_price = None
        self.last_action = 0
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        self.entry_price = None
        self.steps_since_entry = None

        # Pipeline setup
        self.pipeline = PipelineOrchestrator(
            timeframes=["1m", "3m", "5m", "15m", "1h", "4h", "1d"],
            candle_limits={"1m": 60, "3m": 60, "5m": 60, "15m": 60,
                           "1h": 60, "4h": 60, "1d": 60},
            preload_days=60
        )

        # Preload
        while not self.pipeline.preload_done and self.current_index < len(self.meta_df):
            bar = self.meta_df.iloc[self.current_index].to_dict()
            self.pipeline.process_new_candle(bar)
            self.current_index += 1

        self.start_index = self.current_index
        logger.info(f"[ENV INIT] Preload complete. Start index set to {self.start_index}")

        self.observation_space = Dict({
            "images": Box(low=0, high=1, shape=(7, 3, 640, 640), dtype=np.float32),
            "features": Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32),
        })
        self.action_space = gym.spaces.Discrete(5)

    def reset(self, seed=None, options=None):
        logger.debug("[RESET] Called")

        self.current_index = self.start_index
        self.position = 0
        self.last_price = None
        self.last_action = 0
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        self.entry_price = None
        self.steps_since_entry = None

        obs = {
            "images": np.zeros((7, 3, 640, 640), dtype=np.float32),
            "features": np.zeros((6,), dtype=np.float32)
        }
        return obs, {}

    def step(self, action):
        if action not in [0, 1, 2, 3, 4]:
            raise ValueError(f"Invalid action {action}")

        if self.current_index >= len(self.meta_df):
            logger.warning("[STEP] End of data reached.")
            return self.reset(), 0.0, True, False, {}

        row = self.meta_df.iloc[self.current_index].to_dict()
        close_price = row["close"]

        if self.last_price is None:
            self.last_price = close_price

        if self.position == 1:
            self.unrealized_pnl = close_price - self.entry_price
        elif self.position == -1:
            self.unrealized_pnl = self.entry_price - close_price
        else:
            self.unrealized_pnl = 0.0

        reward = 0.0
        invalid_action = False

        # --- ACTION LOGIC ---
        if action == 1 and self.position == 0:
            self.position = 1
            self.entry_price = close_price
            self.steps_since_entry = 0
            reward = 1.0
        elif action == 2 and self.position == 0:
            self.position = -1
            self.entry_price = close_price
            self.steps_since_entry = 0
            reward = 1.0
        elif action == 3 and self.position == 1:
            reward = close_price - self.entry_price
            self.realized_pnl += reward
            self.position = 0
            self.entry_price = None
            self.steps_since_entry = None
        elif action == 4 and self.position == -1:
            reward = self.entry_price - close_price
            self.realized_pnl += reward
            self.position = 0
            self.entry_price = None
            self.steps_since_entry = None
        elif action == 0:
            reward = 0.0
        else:
            invalid_action = True
            reward = -1.0

        if self.position != 0 and self.steps_since_entry is not None:
            self.steps_since_entry += 1

        self.last_action = action

        # --- Generate Observation ---
        output = self.pipeline.process_new_candle(row)
        if output is None:
            # logger.debug("[OBS] No output from pipeline. Returning zero obs.")
            obs = {
                "images": np.zeros((7, 3, 640, 640), dtype=np.float32),
                "features": np.zeros((6,), dtype=np.float32),
            }
        else:
            img_tensor = output["images"]
            # logger.info(f"[OBS] Raw image tensor shape: {tuple(img_tensor.shape)}")

            if img_tensor.dim() == 5:
                img_tensor = img_tensor.squeeze(0)
                # logger.info(f"[OBS] Squeezed image tensor to: {tuple(img_tensor.shape)}")

            # Log if image shape is incorrect
            # if img_tensor.shape != (7, 3, 640, 640):
            #     logger.warning(f"[OBS] Unexpected image shape: {img_tensor.shape}")

            obs = {
                "images": img_tensor.detach().cpu().numpy(),
                "features": np.array([
                    self.realized_pnl,
                    self.unrealized_pnl,
                    self.position,
                    self.last_action,
                    self.steps_since_entry or 0,
                    reward
                ], dtype=np.float32)
            }

        self.current_index += 1
        terminated = self.current_index >= len(self.meta_df)
        truncated = False
        return obs, reward, terminated, truncated, {}

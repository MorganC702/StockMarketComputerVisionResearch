# test_setup.py
import os
import numpy as np
import pandas as pd
from PIL import Image

from envs.trading_env import TradingEnv
from models.yolo_encoder import YOLOEncoder
from models.policy import YOLOResNetExtractor

import torch
from stable_baselines3 import PPO

# ------------------------
# 1. Create fake image dataset
# ------------------------
os.makedirs("fake_images", exist_ok=True)

num_samples = 5
paths = []
timestamps = pd.date_range("2023-01-01", periods=num_samples, freq="T")

for i in range(num_samples):
    path = f"fake_images/img_{i}.png"
    # random noise image
    arr = (np.random.rand(640, 640, 3) * 255).astype(np.uint8)
    Image.fromarray(arr).save(path)
    paths.append(path)

# ------------------------
# 2. Build fake meta_df
# ------------------------
meta_df = pd.DataFrame({
    "timestamp": timestamps,
    "path": paths,
    "close": np.linspace(100, 110, num_samples)  # linearly increasing prices
})

print("Meta_df sample:")
print(meta_df.head())

# ------------------------
# 3. Build YOLO encoder + env
# ------------------------
yolo_encoder = YOLOEncoder("./best.pt")   # expects weights, can be dummy if just testing hooks
env = TradingEnv(meta_df)

# ------------------------
# 4. Step through env
# ------------------------
obs, _ = env.reset()
print("\nObs keys:", obs.keys())
print("Image shape:", obs["image"].shape)
print("Features:", obs["features"])

obs, reward, done, _, info = env.step(env.action_space.sample())
print("\nStep result:")
print("Reward:", reward)
print("Done:", done)
print("Info:", info)

# ------------------------
# 5. Test PPO wiring (small run)
# ------------------------
policy_kwargs = dict(
    features_extractor_class=YOLOResNetExtractor,
    features_extractor_kwargs=dict(yolo_encoder=yolo_encoder, features_dim=512+3),
)

model = PPO(
    policy="MultiInputPolicy",
    env=env,
    policy_kwargs=policy_kwargs,
    verbose=1,
    device="cpu"
)

print("\nTraining PPO for 2 steps just to check wiring...")
model.learn(total_timesteps=2)

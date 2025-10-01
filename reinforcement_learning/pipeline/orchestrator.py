import torch
from collections import deque
from tqdm import tqdm
import os
import pandas as pd

from pipeline.aggregator import TimeframeAggregator
from pipeline.feature_extractor import FeatureExtractor
from pipeline.generate_image import ImageGenerator
from pipeline.image_stack_builder import ImageStackBuilder
from pipeline.zone_engine import ZoneEngine 


class PipelineOrchestrator:
    def __init__(self, timeframes, candle_limits, preload_days=60):
        self.buffer = deque(maxlen=(preload_days * 24 * 60))
        self.aggregator = TimeframeAggregator(self.buffer)
        self.zone_engine = ZoneEngine(timeframes)
        self.image_generator = ImageGenerator(candle_limits)
        self.extractor = FeatureExtractor()
        self.image_stack_builder = ImageStackBuilder(timeframes)
        self.timeframes = timeframes

        self.preload_minutes = preload_days * 24 * 60
        self.preload_done = False
        self.pbar = tqdm(total=self.preload_minutes, desc=f"Preloading {preload_days} days", unit="min")

        self.cached_visible_zones = {tf: [] for tf in self.timeframes}

    def process_new_candle(self, bar: dict):
        if 'timestamp' not in bar:
            raise KeyError(f"Bar missing 'timestamp' key: {bar}")

        self.buffer.append(bar)

        # --- Preload Guard ---
        if not self.preload_done:
            self.pbar.update(1)
            if len(self.buffer) >= self.preload_minutes:
                self.preload_done = True
                self.pbar.close()
                print(f"[INFO] Preload complete after {self.preload_minutes} minutes. Starting image generation.")
            if not self.preload_done:
                return None

        # --- Resample all timeframes ---
        tf_dfs = self.aggregator.resample_all(self.timeframes)
        image_paths = {}
        current_ts = pd.to_datetime(bar["timestamp"])

        for tf in self.timeframes:
            tf_df = tf_dfs.get(tf)
            if tf_df is None or tf_df.empty:
                continue

            last_candle = tf_df.tail(1)
            ts = pd.to_datetime(last_candle["timestamp"].values[0])
            tf_minutes = self.zone_engine._parse_tf_to_minutes(tf)
            aligned_ts = current_ts.floor(f"{tf_minutes}min")

            if ts == aligned_ts and current_ts == aligned_ts:
                self.zone_engine.update(tf, last_candle.iloc[0].to_dict())
                self.cached_visible_zones[tf] = self.zone_engine.get_visible_zones(tf)

            zones = self.cached_visible_zones[tf]

            window_len = self.image_generator.candle_limits[tf]
            tf_df_window = tf_df.tail(window_len).copy()

            image_dir = os.path.join("./data/img_dataset", tf, "images")
            os.makedirs(image_dir, exist_ok=True)

            filename = f"{str(current_ts).replace(':', '_').replace('/', '-')}.png"
            image_path = os.path.join(image_dir, filename)

            self.image_generator.generate_image(tf, tf_df_window, zones, image_path)
            image_paths[tf] = image_path

        if len(image_paths) != len(self.timeframes):
            print(f"[WARN] Not all timeframes produced images. Skipping feature extraction.")
            return None

        # --- Build tensor stack ---
        image_tensor = self.image_stack_builder.build_stack(image_paths)  # [7,3,H,W]

        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)
        elif image_tensor.dim() != 4:
            raise ValueError(f"Unexpected image tensor shape: {image_tensor.shape}")

        return {
            "images": image_tensor  # [7,3,H,W]
            # You will attach real 'features' (balance, etc.) inside TradingEnv
        }

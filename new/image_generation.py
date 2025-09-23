import os
from collections import deque
from datetime import timedelta

import pandas as pd
from PIL import Image, ImageDraw


class MultiTimeframeImageGen:
    def __init__(
            self,
            timeframes,
            base_tf="1m",
            window_size=1 * 24 * 60,
            history_days=30,
            output_root=None,
            image_size=(640, 640),
            dot_radius=2,
            max_trail_length=100,
            price_scale=1.0,
            show_trail=True,
            normalize=True   # <--- lock-to-center mode
        ):
        
        self.base_tf = base_tf
        self.timeframes = timeframes
        self.window_size = window_size
        self.history_days = history_days

        # rendering params
        self.output_root = output_root
        self.image_size = image_size
        self.dot_radius = dot_radius
        self.max_trail_length = max_trail_length
        self.price_scale = price_scale
        self.show_trail = show_trail
        self.normalize = normalize

        # frame counters (sequential filenames)
        self.frame_counters = {tf: 0 for tf in timeframes}

        # data buffers
        self.base_buffer = deque(maxlen=window_size)
        self.buffers = {tf: deque(maxlen=window_size) for tf in timeframes}

        # stacks
        self.high_break_levels = {tf: [] for tf in timeframes}
        self.low_break_levels = {tf: [] for tf in timeframes}

        # history
        self.historical_events = {tf: deque() for tf in timeframes}
        self.global_history = deque()

        # timeframe map
        self.tf_to_minutes = {
            "1m": 1, "3m": 3, "5m": 5,
            "15m": 15, "1h": 60,
            "4h": 240, "1d": 1440
        }

    # ------------------ main feed ------------------

    def get_last(self, row: pd.Series, preload: bool = False):
        """Feed one new bar. If preload=True, skip rendering images."""
        self.base_buffer.append(row)
        self._process_tf(row, "1m")

        for tf in self.timeframes:
            if tf == "1m":
                continue
            multiple = self.tf_to_minutes[tf]
            ts = row.name
            if self._is_boundary(ts, multiple):
                chunk = list(self.base_buffer)[-multiple:]
                if len(chunk) < multiple:
                    continue
                df = pd.DataFrame(chunk)
                df.index = [r.name for r in chunk]
                agg_row = pd.Series({
                    "Open": df["Open"].iloc[0],
                    "High": df["High"].max(),
                    "Low": df["Low"].min(),
                    "Close": df["Close"].iloc[-1],
                    "Volume": df["Volume"].sum() if "Volume" in df else None,
                }, name=ts)
                self._process_tf(agg_row, tf)

        if not preload and self.output_root is not None:
            self._render_images(row.name)

    # ------------------ helpers ------------------

    def _is_boundary(self, ts, multiple):
        if multiple < 60:
            return ts.minute % multiple == 0 and ts.second == 0
        elif multiple < 1440:
            hours = multiple // 60
            return ts.minute == 0 and ts.hour % hours == 0 and ts.second == 0
        else:  # daily
            return ts.hour == 0 and ts.minute == 0 and ts.second == 0

    def _process_tf(self, row, tf):
        buf = self.buffers[tf]
        buf.append(row)
        if len(buf) < 3:
            return

        df = pd.DataFrame(buf)
        levels_df = self._detect_break_levels(df)
        if not levels_df.empty:
            last_lvl = levels_df.iloc[-1]
            entry = {
                "level_price": last_lvl["level_price"],
                "level_time": last_lvl["level_time"],
                "type": last_lvl["type"],
                "active": True
            }
            if entry["type"] == "high_break":
                if not self.high_break_levels[tf] or entry["level_time"] > self.high_break_levels[tf][-1]["level_time"]:
                    self.high_break_levels[tf].append(entry)
            else:
                if not self.low_break_levels[tf] or entry["level_time"] > self.low_break_levels[tf][-1]["level_time"]:
                    self.low_break_levels[tf].append(entry)
        self._update_stacks(row, tf)

    def _detect_break_levels(self, df):
        directions = df["Close"].sub(df["Open"]).apply(
            lambda x: 1 if x > 0 else -1 if x < 0 else 0
        )
        swing_levels, i, n = [], 0, len(df)
        current_swing, extreme_price, extreme_time, opposite_streak = None, None, None, 0

        while i < n:
            d = directions.iloc[i]
            row = df.iloc[i]
            time = df.index[i]
            if d == 0:
                i += 1
                continue
            if current_swing is None:
                current_swing = 'high_break' if d == 1 else 'low_break'
                extreme_price = row["High"] if d == 1 else row["Low"]
                extreme_time = time
                i += 1
                continue
            if ((current_swing == 'high_break' and d == 1) or
                (current_swing == 'low_break' and d == -1)):
                if current_swing == 'high_break' and row["High"] > extreme_price:
                    extreme_price, extreme_time = row["High"], time
                elif current_swing == 'low_break' and row["Low"] < extreme_price:
                    extreme_price, extreme_time = row["Low"], time
                opposite_streak = 0
            else:
                opposite_streak += 1
                if opposite_streak >= 2:
                    swing_levels.append({
                        "type": current_swing,
                        "level_price": extreme_price,
                        "level_time": extreme_time
                    })
                    current_swing = 'low_break' if current_swing == 'high_break' else 'high_break'
                    extreme_price = row["High"] if current_swing == 'high_break' else row["Low"]
                    extreme_time, opposite_streak = time, 0
            i += 1
        return pd.DataFrame(swing_levels)

    def _update_stacks(self, row, tf):
        timestamp, close = row.name, row["Close"]

        still_active_highs = []
        for lvl in self.high_break_levels[tf]:
            if close > lvl["level_price"]:
                lvl["break_time"], lvl["active"] = timestamp, False
                self._add_to_history(tf, lvl)
            else:
                still_active_highs.append(lvl)
        self.high_break_levels[tf] = still_active_highs

        still_active_lows = []
        for lvl in self.low_break_levels[tf]:
            if close < lvl["level_price"]:
                lvl["break_time"], lvl["active"] = timestamp, False
                self._add_to_history(tf, lvl)
            else:
                still_active_lows.append(lvl)
        self.low_break_levels[tf] = still_active_lows

    def _add_to_history(self, tf, event):
        tagged = {**event, "timeframe": tf}
        self.historical_events[tf].append(tagged)
        self._prune_history(self.historical_events[tf], tagged["break_time"])
        self.global_history.append(tagged)
        self._prune_history(self.global_history, tagged["break_time"])

    def _prune_history(self, queue, now_time):
        cutoff = now_time - timedelta(days=self.history_days)
        while queue and queue[0]["break_time"] < cutoff:
            queue.popleft()

    # ------------------ rendering ------------------

    def _render_images(self, ts):
        base_df = pd.DataFrame(self.buffers[self.base_tf])
        if base_df.empty or ts not in base_df.index:
            return

        # normalization setup
        if self.normalize:
            window_high = base_df["High"].max()
            window_low = base_df["Low"].min()
            price_range = max(window_high - window_low, 1e-9)
            price_now_norm = (base_df.loc[ts]["Close"] - window_low) / price_range
            trail_df = base_df.loc[:ts].iloc[-self.max_trail_length:-1].copy()
            trail_df["Close"] = (trail_df["Close"] - window_low) / price_range
        else:
            price_now = base_df.loc[ts]["Close"]
            trail_df = base_df.loc[:ts].iloc[-self.max_trail_length:-1]
            half_viewport = self.image_size[1] / (2 * self.price_scale)
            price_min, price_max = price_now - half_viewport, price_now + half_viewport

        for tf in self.timeframes:
            self.frame_counters[tf] += 1
            frame_num = self.frame_counters[tf]
            tf_dir = os.path.join(self.output_root, tf, "images")
            lbl_dir = os.path.join(self.output_root, tf, "labels")
            os.makedirs(tf_dir, exist_ok=True)
            os.makedirs(lbl_dir, exist_ok=True)

            out_file = os.path.join(tf_dir, f"frame_{frame_num:05d}.png")
            lbl_file = os.path.join(lbl_dir, f"frame_{frame_num:05d}.txt")

            img = Image.new("RGB", self.image_size, (255, 255, 255))
            draw = ImageDraw.Draw(img)

            x_center = self.image_size[0] // 2
            y_center = self.image_size[1] // 2
            yolo_labels = []

            # active levels
            levels_df = pd.DataFrame(self.get_active_levels(tf)["high"] +
                                     self.get_active_levels(tf)["low"])
            events_df = pd.DataFrame(self.get_historical_events(tf))

            if not levels_df.empty:
                formed = levels_df[levels_df["level_time"] <= ts]
                for _, row in formed.iterrows():
                    lvl_price, lvl_type = row["level_price"], row["type"]

                    if not events_df.empty:
                        match = events_df[
                            (events_df["level_price"] == lvl_price) &
                            (events_df["type"] == lvl_type) &
                            (events_df["break_time"] <= ts)
                        ]
                        if not match.empty:
                            continue

                    if self.normalize:
                        lvl_price_norm = (lvl_price - window_low) / price_range
                        dy = (lvl_price_norm - price_now_norm) * self.price_scale * self.image_size[1]
                        y = int(y_center - dy)
                    else:
                        if not (price_min <= lvl_price <= price_max):
                            continue
                        y = y_center - (lvl_price - price_now) * self.price_scale

                    try:
                        lvl_idx = base_df.index.get_loc(row["level_time"])
                    except KeyError:
                        continue

                    frames_since = base_df.index.get_loc(ts) - lvl_idx
                    if frames_since < 0:
                        continue

                    x_start = max(x_center - frames_since * 2 * self.dot_radius, 0)
                    color = (255, 0, 0) if lvl_type == "high_break" else (0, 200, 0)
                    draw.line([(x_start, y), (self.image_size[0], y)], fill=color, width=2)

                    xc = 0.5
                    yc = y / self.image_size[1]
                    bw = 1.0
                    bh = 2 * self.dot_radius / self.image_size[1]
                    cls = 0 if lvl_type == "high_break" else 1
                    yolo_labels.append((cls, xc, yc, bw, bh))

            # trail
            if self.show_trail:
                for j, (_, row) in enumerate(trail_df.iterrows()):
                    if self.normalize:
                        t_price_norm = row["Close"]
                        dy = (t_price_norm - price_now_norm) * self.price_scale * self.image_size[1]
                        cy = int(y_center - dy)
                    else:
                        t_price = row["Close"]
                        dy = (t_price - price_now) * self.price_scale
                        cy = y_center - dy

                    cx = x_center - (len(trail_df) - j) * 2 * self.dot_radius
                    gray_val = int(255 * (1 - (j + 1) / max(1, len(trail_df))))
                    draw.ellipse([cx - self.dot_radius, cy - self.dot_radius,
                                  cx + self.dot_radius, cy + self.dot_radius],
                                 fill=(gray_val, gray_val, gray_val))

            # center dot always locked
            cy = y_center
            draw.ellipse([x_center - self.dot_radius, cy - self.dot_radius,
                          x_center + self.dot_radius, cy + self.dot_radius],
                         fill=(0, 0, 255))

            xc = x_center / self.image_size[0]
            yc = cy / self.image_size[1]
            bw = 2 * self.dot_radius / self.image_size[0]
            bh = 2 * self.dot_radius / self.image_size[1]
            yolo_labels.append((2, xc, yc, bw, bh))

            img.save(out_file)
            with open(lbl_file, "w") as f:
                for cls, xc, yc, bw, bh in yolo_labels:
                    f.write(f"{cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

    # ------------------ public API ------------------

    def get_active_levels(self, tf):
        return {
            "high": list(self.high_break_levels[tf]),
            "low": list(self.low_break_levels[tf])
        }

    def get_historical_events(self, tf):
        return list(self.historical_events[tf])

    def get_global_history(self):
        return list(self.global_history)

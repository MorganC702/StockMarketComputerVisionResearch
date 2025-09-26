import os
from collections import deque
from datetime import timedelta

import pandas as pd
import numpy as np
import cv2


class MultiTimeframeImageGen:
    def __init__(
        self,
        timeframes,
        base_tf="1m",
        output_root=None,
        image_size=(640, 640),
        window_sizes=None,   # dict like {"1m": 60*60, "1d": 90}
        right_padding=0.2    # fraction of chart width reserved as padding
    ):
        self.base_tf = base_tf
        self.timeframes = timeframes
        self.output_root = output_root
        self.image_size = image_size
        self.right_padding = right_padding

        # windows (bars) per timeframe
        self.window_sizes = window_sizes or {tf: 500 for tf in timeframes}

        # buffers per timeframe
        self.buffers = {tf: deque(maxlen=self.window_sizes[tf]) for tf in timeframes}
        self.current_bars = {tf: None for tf in timeframes}  # hold in-progress candles

        # stacks
        self.high_break_levels = {tf: [] for tf in timeframes}
        self.low_break_levels = {tf: [] for tf in timeframes}

        # history
        self.historical_events = {tf: deque() for tf in timeframes}
        self.global_history = deque()

        # frame counters
        self.frame_counters = {tf: 0 for tf in timeframes}

        # timeframe map (minutes per candle)
        self.tf_to_minutes = {
            "1m": 1, "3m": 3, "5m": 5,
            "15m": 15, "1h": 60,
            "4h": 240, "1d": 1440
        }

        # --- NEW: meta file path ---
        self.meta_path = os.path.join(self.output_root, "metadata.csv") if self.output_root else None
        if self.meta_path and os.path.exists(self.meta_path):
            os.remove(self.meta_path)  # start fresh each run

    # ------------------ main feed ------------------

    def get_last(self, row: pd.Series, preload: bool = False):
        """Feed one new 1m bar and update all TFs."""
        self._update_tf(row, "1m", preload)

        # aggregate higher TFs
        for tf in self.timeframes:
            if tf == "1m":
                continue
            self._update_tf(row, tf, preload)

        # render all charts every minute (reflects partial candle formation)
        if not preload and self.output_root is not None:
            self._render_images(row.name)

    # ------------------ timeframe update ------------------

    def _update_tf(self, row, tf, preload):
        multiple = self.tf_to_minutes[tf]
        ts = row.name

        # start new candle if none yet
        if self.current_bars[tf] is None:
            self.current_bars[tf] = pd.Series({
                "Open": row["Open"],
                "High": row["High"],
                "Low": row["Low"],
                "Close": row["Close"],
                "Volume": row.get("Volume", 0)
            }, name=ts)
            self.buffers[tf].append(self.current_bars[tf])
            return

        # check if boundary crossed -> finalize last bar, start new one
        last_bar_time = self.current_bars[tf].name
        minutes_passed = int((ts - last_bar_time).total_seconds() // 60)
        if minutes_passed >= multiple:
            self.current_bars[tf] = pd.Series({
                "Open": row["Open"],
                "High": row["High"],
                "Low": row["Low"],
                "Close": row["Close"],
                "Volume": row.get("Volume", 0)
            }, name=ts)
            self.buffers[tf].append(self.current_bars[tf])
        else:
            self.current_bars[tf]["High"] = max(self.current_bars[tf]["High"], row["High"])
            self.current_bars[tf]["Low"] = min(self.current_bars[tf]["Low"], row["Low"])
            self.current_bars[tf]["Close"] = row["Close"]
            if "Volume" in row:
                self.current_bars[tf]["Volume"] += row["Volume"]
            self.buffers[tf][-1] = self.current_bars[tf]

        # process levels
        buf = pd.DataFrame(self.buffers[tf])
        if len(buf) >= 3:
            levels_df = self._detect_break_levels(buf)
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

    # ------------------ level detection ------------------

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
        cutoff = now_time - timedelta(days=365)
        while queue and queue[0]["break_time"] < cutoff:
            queue.popleft()

    # ------------------ rendering ------------------

    def _render_images(self, ts):
        row_record = {"timestamp": ts}

        for tf in self.timeframes:
            buf = pd.DataFrame(self.buffers[tf])

            # include unfinished current candle
            cur_bar = self.current_bars.get(tf)
            if cur_bar is not None:
                if buf.empty or cur_bar.name != buf.index[-1]:
                    buf = pd.concat([buf, pd.DataFrame([cur_bar])])

            if buf.empty:
                continue

            window = buf.iloc[-self.window_sizes[tf]:]
            if window.empty:
                continue

            hi, lo = window["High"].max(), window["Low"].min()
            price_range = hi - lo if hi > lo else 1e-9

            # padding (1% each side)
            pad_x = int(self.image_size[0] * 0.01)
            pad_y = int(self.image_size[1] * 0.01)
            draw_width = self.image_size[0] - 2 * pad_x
            draw_height = self.image_size[1] - 2 * pad_y

            def price_to_y(p):
                return pad_y + int((hi - p) / price_range * (draw_height - 1))

            n = len(window)
            usable_w = int(draw_width * (1 - self.right_padding))
            step_x = usable_w / max(1, n - 1)

            def index_to_x(i):
                return pad_x + int(i * step_x)

            # create white background
            img = np.ones((self.image_size[1], self.image_size[0], 3), dtype=np.uint8) * 255

            # candles
            for i, (_, row) in enumerate(window.iterrows()):
                x = index_to_x(i)
                y_open, y_close = price_to_y(row["Open"]), price_to_y(row["Close"])
                y_high, y_low = price_to_y(row["High"]), price_to_y(row["Low"])
                color = (0, 0, 0) if row["Close"] >= row["Open"] else (128, 128, 128)
                # wick
                cv2.line(img, (x, y_high), (x, y_low), color, 2)
                # body
                cv2.rectangle(img, (x - 2, y_open), (x + 2, y_close), color, -1)

            # dashed price line (blue)
            close_price = window.iloc[-1]["Close"]
            y_close = price_to_y(close_price)
            dash_len = 10
            for x in range(pad_x, self.image_size[0] - pad_x, dash_len * 2):
                cv2.line(img, (x, y_close), (x + dash_len, y_close), (255, 0, 0), 1)

            # --- YOLO labels ---
            yolo_labels = []

            def make_yolo_box(x0, y0, x1, y1, class_id):
                pad_w = int(self.image_size[0] * 0.01)
                pad_h = int(self.image_size[1] * 0.01)
                x0 = max(0, x0 - pad_w)
                y0 = max(0, y0 - pad_h)
                x1 = min(self.image_size[0], x1 + pad_w)
                y1 = min(self.image_size[1], y1 + pad_h)

                x_center = (x0 + x1) / 2 / self.image_size[0]
                y_center = (y0 + y1) / 2 / self.image_size[1]
                w_norm = (x1 - x0) / self.image_size[0]
                h_norm = (y1 - y0) / self.image_size[1]

                return f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}"

            # price line box
            box_half_h = max(6, int(self.image_size[1] * 0.01))
            y0 = max(y_close - box_half_h, 0)
            y1 = min(y_close + box_half_h, self.image_size[1])
            x0, x1 = pad_x, self.image_size[0] - pad_x
            yolo_labels.append(make_yolo_box(x0, y0, x1, y1, class_id=0))

            # levels
            levels = self.high_break_levels[tf] + self.low_break_levels[tf]
            for lvl in levels:
                if not lvl["active"]:
                    continue
                y = price_to_y(lvl["level_price"])
                box_half_h = max(6, int(self.image_size[1] * 0.01))
                y0 = max(y - box_half_h, 0)
                y1 = min(y + box_half_h, self.image_size[1])

                try:
                    locs = window.index.get_loc(lvl["level_time"])
                    if isinstance(locs, slice):
                        lvl_idx = locs.start
                    elif isinstance(locs, (list, pd.Series, pd.Index)):
                        lvl_idx = locs[0]
                    else:
                        lvl_idx = locs
                except KeyError:
                    continue

                x_start = index_to_x(lvl_idx)
                x_end = self.image_size[0] - pad_x
                class_id = 1 if lvl["type"] == "high_break" else 2
                yolo_labels.append(make_yolo_box(x_start, y0, x_end, y1, class_id))

                # draw line
                color = (0, 0, 255) if lvl["type"] == "high_break" else (0, 200, 0)
                cv2.line(img, (x_start, y), (x_end, y), color, 2)

            # save image + labels
            self.frame_counters[tf] += 1
            frame_num = self.frame_counters[tf]

            tf_img_dir = os.path.join(self.output_root, tf, "images")
            tf_lbl_dir = os.path.join(self.output_root, tf, "labels")
            os.makedirs(tf_img_dir, exist_ok=True)
            os.makedirs(tf_lbl_dir, exist_ok=True)

            img_path = os.path.join(tf_img_dir, f"frame_{frame_num:05d}.png")
            lbl_path = os.path.join(tf_lbl_dir, f"frame_{frame_num:05d}.txt")

            cv2.imwrite(img_path, img)
            with open(lbl_path, "w") as f:
                f.write("\n".join(yolo_labels))

            # --- NEW: record metadata ---
            row_record["close"] = close_price
            row_record[tf] = img_path

        # append row immediately to CSV
        if self.meta_path and len(row_record) > 2:
            df = pd.DataFrame([row_record])
            write_header = not os.path.exists(self.meta_path)
            df.to_csv(self.meta_path, mode="a", header=write_header, index=False)

    # ------------------ public API ------------------

    def get_active_levels(self, tf):
        return {"high": list(self.high_break_levels[tf]), "low": list(self.low_break_levels[tf])}

    def get_historical_events(self, tf):
        return list(self.historical_events[tf])

    def get_global_history(self):
        return list(self.global_history)

import numpy as np
import pandas as pd
import cv2
import os

def draw_faded_rectangle(img, pt1, pt2, color, alpha):
    """Draws a transparent rectangle onto img with alpha blending."""
    overlay = img.copy()
    cv2.rectangle(overlay, pt1, pt2, color, -1)
    return cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

def generate_zone_image(
    window: pd.DataFrame,
    visible_zones: list,
    save_path: str,
    image_size=(640, 640),
    right_padding_bars: int = 30,
    save_yolo_labels: bool = True,
):
    width, height = image_size
    pad_x = int(width * 0.01)
    pad_y = int(height * 0.01)

    usable_width = width - 2 * pad_x
    usable_height = height - 2 * pad_y

    # --- Background (pure white) ---
    img = np.ones((height, width, 3), dtype=np.uint8) * 255

    # --- Center Y-axis around last close price ---
    center_price = window.iloc[-1]["close"]
    hi = window["high"].max()
    lo = window["low"].min()
    max_offset = max(abs(hi - center_price), abs(lo - center_price))

    hi = center_price + max_offset
    lo = center_price - max_offset
    price_range = hi - lo if hi != lo else 1e-6

    def price_to_y(p):
        return pad_y + int((hi - p) / price_range * (usable_height - 1))

    # --- Scaling for time (X axis) ---
    n = len(window)
    usable_bars = n - 1 + right_padding_bars
    step_x = usable_width / max(usable_bars, 1)

    def index_to_x(i):
        return pad_x + int(i * step_x)

    # --- Prepare YOLO label output ---
    yolo_labels = []

    def make_yolo_box(x0, y0, x1, y1, class_id):
        x_center = (x0 + x1) / 2 / width
        y_center = (y0 + y1) / 2 / height
        w_norm = (x1 - x0) / width
        h_norm = (y1 - y0) / height
        return f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}"

    # --- Zones (support/resistance) ---
    for zone in visible_zones:
        base_color = (0, 255, 0) if zone["zone_type"] == -1 else (0, 0, 255)
        class_id = 1 if zone["zone_type"] == -1 else 2  # 1=Demand (support), 2=Supply (resistance)

        y0 = price_to_y(zone["high"])
        y1 = price_to_y(zone["low"])

        match_pivot = window[window["timestamp"] >= zone["pivot_timestamp"]]
        if match_pivot.empty:
            continue
        pivot_idx = window.index.get_loc(match_pivot.index[0])
        x0 = index_to_x(pivot_idx)

        if zone.get("zone_broken"):
            x1 = index_to_x(pivot_idx + right_padding_bars)
            alpha = 0.05
        else:
            x1 = index_to_x(n - 1 + right_padding_bars)
            alpha = 0.4

        # Draw rectangle
        img = draw_faded_rectangle(img, (x0, y0), (x1, y1), base_color, alpha)

        # Save YOLO box if active
        if save_yolo_labels and not zone.get("zone_broken", False):
            yolo_labels.append(make_yolo_box(x0, y0, x1, y1, class_id))

    # --- Candles ---
    for i, (_, row) in enumerate(window.iterrows()):
        x = index_to_x(i)
        y_open, y_close = price_to_y(row["open"]), price_to_y(row["close"])
        y_high, y_low = price_to_y(row["high"]), price_to_y(row["low"])
        color = (0, 0, 0) if row["close"] >= row["open"] else (128, 128, 128)

        cv2.line(img, (x, y_high), (x, y_low), color, 1)
        cv2.rectangle(img, (x - 2, y_open), (x + 2, y_close), color, -1)

    # --- Last price dashed line ---
    close_price = window.iloc[-1]["close"]
    y = price_to_y(close_price)
    dash_len = 10
    for x in range(pad_x, width - pad_x, dash_len * 2):
        cv2.line(img, (x, y), (x + dash_len, y), (255, 0, 0), 1)

    # Save YOLO label for price line (class 0)
    if save_yolo_labels:
        box_thickness = 15  # half-height in pixels → total ~30px tall
        y0_price = max(0, y - box_thickness)
        y1_price = min(height - 1, y + box_thickness)
        yolo_labels.append(make_yolo_box(pad_x, y0_price, width - pad_x, y1_price, 0))


    # --- Save image ---
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, img)

    # --- Save YOLO label ---
    if save_yolo_labels and yolo_labels:
        label_path = save_path.replace("/images/", "/labels/").replace(".png", ".txt")
        os.makedirs(os.path.dirname(label_path), exist_ok=True)
        with open(label_path, "w") as f:
            f.write("\n".join(yolo_labels))


class ImageGenerator:
    def __init__(self, candle_limits: dict, image_size=(640, 640)):
        self.candle_limits = candle_limits
        self.image_size = image_size

    def generate_image(self, tf: str, df: pd.DataFrame, zones: list, save_path: str):
        candle_limit = self.candle_limits.get(tf, 60)
        window = df.tail(candle_limit)
        generate_zone_image(window, zones, save_path, image_size=self.image_size, save_yolo_labels=True)

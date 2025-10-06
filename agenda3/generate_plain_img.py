import numpy as np
import pandas as pd
import cv2
import os

def draw_candle(img, x, row, price_to_y, color):
    """Draw a single candlestick on the image."""
    y_open, y_close = price_to_y(row["open"]), price_to_y(row["close"])
    y_high, y_low = price_to_y(row["high"]), price_to_y(row["low"])

    # Wick
    cv2.line(img, (x, y_high), (x, y_low), color, 1)
    # Body
    cv2.rectangle(img, (x - 2, y_open), (x + 2, y_close), color, -1)


def generate_plain_image(
    window: pd.DataFrame,
    save_path: str,
    image_size=(640, 640),
    save_yolo_labels: bool = True,
):
    width, height = image_size
    pad_x = int(width * 0.02)
    pad_y = int(height * 0.02)

    usable_width = width - 2 * pad_x
    usable_height = height - 2 * pad_y

    # Background white
    img = np.ones((height, width, 3), dtype=np.uint8) * 255

    # Price scaling (high/low range)
    hi = window["high"].max()
    lo = window["low"].min()
    price_range = hi - lo if hi != lo else 1e-6

    def price_to_y(p):
        return pad_y + int((hi - p) / price_range * (usable_height - 1))

    # Time scaling
    n = len(window)
    step_x = usable_width / max(n - 1, 1)

    def index_to_x(i):
        return pad_x + int(i * step_x)

    # --- Draw candles ---
    yolo_labels = []
    for i, (_, row) in enumerate(window.iterrows()):
        x = index_to_x(i)
        color = (0, 0, 0) if row["close"] >= row["open"] else (128, 128, 128)
        draw_candle(img, x, row, price_to_y, color)

        if save_yolo_labels:
            # YOLO box around the candle body
            y0 = min(price_to_y(row["open"]), price_to_y(row["close"]))
            y1 = max(price_to_y(row["open"]), price_to_y(row["close"]))
            x0 = x - 2
            x1 = x + 2

            # Normalize
            x_center = (x0 + x1) / 2 / width
            y_center = (y0 + y1) / 2 / height
            w_norm = (x1 - x0) / width
            h_norm = (y1 - y0) / height

            yolo_labels.append(f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")

    # --- Save image ---
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, img)

    # --- Save YOLO labels ---
    if save_yolo_labels and yolo_labels:
        label_path = save_path.replace("/images/", "/labels/").replace(".png", ".txt")
        os.makedirs(os.path.dirname(label_path), exist_ok=True)
        with open(label_path, "w") as f:
            f.write("\n".join(yolo_labels))


class ImageGenerator:
    def __init__(self, candle_limits: dict, image_size=(640, 640)):
        self.candle_limits = candle_limits
        self.image_size = image_size

    def generate_image(self, tf: str, df: pd.DataFrame, save_path: str):
        candle_limit = self.candle_limits.get(tf, 60)
        window = df.tail(candle_limit)
        generate_plain_image(window, save_path, image_size=self.image_size, save_yolo_labels=True)

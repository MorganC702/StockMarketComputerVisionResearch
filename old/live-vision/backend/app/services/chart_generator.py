import numpy as np
import cv2
from io import BytesIO

def generate_candles(
    df_window,
    predictions=None,
    H=640, W=640,
    body_width=20,
    wick_width=2,
    pad_px=5
):
    """
    Core renderer: draws candlesticks and optional YOLO-style prediction boxes.
    Returns a numpy image (H,W,3).
    """
    img = np.ones((H, W, 3), dtype=np.uint8) * 255

    # --- Price scaling ---
    min_price = df_window["Low"].min()
    max_price = df_window["High"].max()
    y_span = max_price - min_price

    def y_map(val): 
        return int(H - (val - min_price) / y_span * H)

    # --- Candles ---
    for i, row in enumerate(df_window.itertuples()):
        o, c, h, l = row.Open, row.Close, row.High, row.Low
        color = (0,200,0) if c > o else (0,0,200)

        x_center = int((i + 0.5) * (W / len(df_window)))
        x0 = x_center - body_width // 2
        x1 = x_center + body_width // 2

        y_high, y_low = y_map(h), y_map(l)
        y_open, y_close = y_map(o), y_map(c)

        # Wick
        cv2.line(img, (x_center, y_high), (x_center, y_low), color, wick_width)

        # Body
        cv2.rectangle(img, (x0, min(y_open, y_close)), (x1, max(y_open, y_close)), color, -1)

    # --- Prediction boxes (if provided) ---
    if predictions:
        for p in predictions:
            # assume p = {"x": int, "y": int, "w": int, "h": int, "label": str}
            cv2.rectangle(img,
                          (p["x"], p["y"]),
                          (p["x"] + p["w"], p["y"] + p["h"]),
                          (255,0,0), 2)
            cv2.putText(img,
                        p["label"],
                        (p["x"] + 4, max(p["y"] - 4, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (0,0,0), 1)

    return img


def generate_chart(df_window, predictions=None, **kwargs):
    """
    Returns PNG-encoded image buffer (BytesIO).
    """
    img = generate_candles(df_window, predictions=predictions, **kwargs)
    success, encoded = cv2.imencode(".png", img)
    if not success:
        raise RuntimeError("Failed to encode image to PNG.")
    return BytesIO(encoded.tobytes())

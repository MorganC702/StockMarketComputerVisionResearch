import os
import numpy as np
import cv2
from tqdm import tqdm

def generate(
    df,
    image_dir="../../iteration_002/images",
    label_dir="../../iteration_002/labels",
    window=5,
    H=640, W=640,
    body_width=20,
    wick_width=2,
    label_to_id=None,
    margin_frac=0.05  # 5% whitespace around edges
):
    """
    Generate candlestick images + YOLO labels in pure OpenCV (no matplotlib).
    df must have columns: Open, High, Low, Close, Label (if label_to_id provided).
    Ensures YOLO labels are normalized to [0,1].
    """
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    # dynamic padding for boxes (~1% of image size)
    pad_px_w = int(0.01 * W)
    pad_px_h = int(0.01 * H)

    for idx in tqdm(range(window, len(df)), desc="Generating images"):
        df_window = df.iloc[idx - window:idx]

        # blank white canvas
        img = np.ones((H, W, 3), dtype=np.uint8) * 255

        boxes, class_ids = [], []

        # margins
        y_margin = int(margin_frac * H)
        x_margin = int(margin_frac * W)

        # map price range → vertical pixels inside margin
        min_price = df_window["Low"].min()
        max_price = df_window["High"].max()
        y_span = max_price - min_price

        def y_map(val): 
            return int(
                H - y_margin - (val - min_price) / y_span * (H - 2 * y_margin)
            )  # top=0, bottom=H

        for i, row in enumerate(df_window.itertuples()):
            o, c, h, l = row.Open, row.Close, row.High, row.Low
            color = (0, 200, 0) if c > o else (0, 0, 200)  # green up, red down

            # horizontal placement (spread across width minus margins)
            x_center = int(
                x_margin + (i + 0.5) * ((W - 2 * x_margin) / window)
            )
            x0 = x_center - body_width // 2
            x1 = x_center + body_width // 2

            # vertical positions
            y_high, y_low = y_map(h), y_map(l)
            y_open, y_close = y_map(o), y_map(c)

            # draw wick
            cv2.line(img, (x_center, y_high), (x_center, y_low), color, wick_width)
            # draw body
            cv2.rectangle(img, (x0, min(y_open, y_close)), (x1, max(y_open, y_close)), color, -1)

            # --- YOLO box (candle low→high, with padding) ---
            x_min = x0 - pad_px_w
            x_max = x1 + pad_px_w
            y_min = y_high - pad_px_h
            y_max = y_low + pad_px_h

            # clamp to image boundaries
            x_min = max(0, x_min)
            x_max = min(W - 1, x_max)
            y_min = max(0, y_min)
            y_max = min(H - 1, y_max)

            # normalize to [0,1]
            xc = (x_min + x_max) / 2 / W
            yc = (y_min + y_max) / 2 / H
            bw = (x_max - x_min) / W
            bh = (y_max - y_min) / H

            boxes.append([xc, yc, bw, bh])
            if label_to_id is not None:
                class_ids.append(label_to_id[row.Label])

        # save image
        img_path = os.path.join(image_dir, f"candle_{idx}.png")
        cv2.imwrite(img_path, img)  


        # save YOLO labels
        if label_to_id is not None:
            label_path = os.path.join(label_dir, f"candle_{idx}.txt")
            with open(label_path, "w") as f:
                for (xc, yc, bw, bh), cls in zip(boxes, class_ids):
                    # guard: ensure final numbers are inside [0,1]
                    xc = min(max(xc, 0), 1)
                    yc = min(max(yc, 0), 1)
                    bw = min(max(bw, 0), 1)
                    bh = min(max(bh, 0), 1)
                    f.write(f"{cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

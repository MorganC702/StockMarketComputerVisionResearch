import os
import cv2, numpy as np, pandas as pd, ast
from tqdm import tqdm

def _as_list(x):
    if isinstance(x, list): return x
    if isinstance(x, str):
        try: return list(ast.literal_eval(x))
        except Exception: return []
    return [] if x is None else [x]

def generate(
    df,
    image_dir="images_out",
    window=100,
    zone_extend=20,
    right_padding=0.2,
    active_only=True,
    H=800, W=1200
):
    """
    Generate candlestick+zone images from ZoneLifecycle.

    Args:
        df (pd.DataFrame): must include Open, High, Low, Close, ZoneLifecycle
        image_dir (str): where to save the images
        window (int): number of candles per image
        zone_extend (int): bars to extend active zones into future
        right_padding (float): fraction of extra width on right
        active_only (bool): 
            - True → plot only Active zones
            - False → plot all confirmed zones
        H, W (int): canvas size in pixels
    """
    os.makedirs(image_dir, exist_ok=True)

    for idx in tqdm(range(window, len(df)), desc="Generating zone images"):
        win = df.iloc[idx - window:idx].reset_index(drop=True)

        margin = 50
        W_pad = int(W * (1 + right_padding))
        img = np.ones((H, W_pad, 3), dtype=np.uint8) * 255

        start_pos = idx - window
        end_pos   = idx - 1

        # y-scale
        pmin, pmax = float(win["Low"].min()), float(win["High"].max())
        if pmax == pmin:
            pmax = pmin + 1e-6
        def y(p):
            return int(margin + (pmax - p) * (H - 2*margin) / (pmax - pmin))

        # x-scale
        step = (W_pad - 2*margin) / (window + zone_extend)
        def x_left(pos):   return int(margin + (pos - start_pos) * step)
        def x_right(pos):  return int(margin + (pos - start_pos + 1) * step)
        def x_center(pos): return int(margin + (pos - start_pos) * step + step/2)
        body_w = max(2, int(step*0.6))

        # ---- draw candles ----
        for j, row in win.iterrows():
            o, c, h, l = row.Open, row.Close, row.High, row.Low
            x = x_center(start_pos + j)
            # Black for down candles, Gray for up candles
            col = (0, 0, 0) if c < o else (128, 128, 128)
            cv2.line(img, (x, y(h)), (x, y(l)), col, 1)
            cv2.rectangle(img,
                        (x - body_w // 2, y(max(o, c))),
                        (x + body_w // 2, y(min(o, c))),
                        col, -1)


        # ---- collect lifecycle ----
        lifecycle = {}
        for j in range(idx - window, idx):
            for z in _as_list(df.iloc[j].get("ZoneLifecycle", [])):
                lifecycle[z["id"]] = z  # keep latest

        # ---- draw zones ----
        overlay = img.copy()
        for zid, z in lifecycle.items():
            status = z.get("status", "Pending")
            pivot  = z.get("pivot_idx")
            c_at   = z.get("confirmed_at")

            # skip never confirmed
            if c_at is None:
                continue
            # skip inactive if only active requested
            if active_only and status != "Active":
                continue
            if pivot is None or not (start_pos <= pivot <= end_pos):
                continue

            end_idx = z.get("end_idx")
            if status == "Active" and end_idx is None:
                last_pos = end_pos
                active_at_end = True
            else:
                last_pos = end_idx if end_idx is not None else end_pos
                active_at_end = False

            x_start = x_left(pivot)
            x_end   = x_right(last_pos + zone_extend) if active_at_end else x_right(last_pos)

            zlow, zhigh, ztype = z["low"], z["high"], z["type"]
            if pd.isna(zlow) or pd.isna(zhigh):
                continue

            y0, y1 = y(zlow), y(zhigh)
            top, bot = min(y0, y1), max(y0, y1)
            color = (0,0,200) if ztype == "Resistance" else (0,200,0)

            cv2.rectangle(overlay, (x_start, top), (x_end, bot), color, -1)
            cv2.line(overlay, (x_start, top), (x_end, top), (50,50,50), 1)
            cv2.line(overlay, (x_start, bot), (x_end, bot), (50,50,50), 1)
            cv2.putText(overlay, str(zid),
                        (max(margin, x_start+6), max(top-6, margin+12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (35,35,35), 2, cv2.LINE_AA)

        # blend + frame
        img = cv2.addWeighted(overlay, 0.25, img, 0.75, 0)
        cv2.rectangle(img, (margin, margin), (W_pad - margin, H - margin), (180,180,180), 1)

        # save
        img_path = os.path.join(image_dir, f"zone_{idx}.png")
        cv2.imwrite(img_path, img)

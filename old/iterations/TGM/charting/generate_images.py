import os
import cv2
import numpy as np
import pandas as pd
import ast
from tqdm import tqdm


def _as_list(x):
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        try:
            return list(ast.literal_eval(x))
        except Exception:
            return []
    return [] if x is None else [x]


def draw_gradient(canvas, x_start, x_end, top, bot, color, intensity=50, sigma=0.5):
    """
    Draw a vertical Gaussian gradient (fade from center to edges in Y only).
    """
    H, W, _ = canvas.shape

    # Clamp bounds
    x_start = max(0, x_start)
    x_end   = min(W, x_end)
    top     = max(0, top)
    bot     = min(H, bot)
    if x_start >= x_end or top >= bot:
        return

    zone_h = bot - top
    ys = np.linspace(-1, 1, zone_h)
    gaussian = np.exp(-(ys**2) / (2 * sigma**2))
    gaussian = (gaussian / gaussian.max()) * intensity

    gaussian_2d = np.tile(gaussian[:, None], (1, x_end - x_start))

    for c in range(3):
        if color[c] > 0:
            canvas[top:bot, x_start:x_end, c] = np.clip(
                canvas[top:bot, x_start:x_end, c].astype(np.float32) + gaussian_2d,
                0, 255
            ).astype(np.uint8)


def generate(
    dfs,
    support_dir="support_out",
    resistance_dir="resistance_out",
    recency_dir="recency_out",
    combined_dir="combined_out",   # NEW
    window=500,
    zone_extend=20,
    right_padding=0.2,
    H=800, W=1200,
    tf_strength=None,
    sigma=0.5
):
    os.makedirs(support_dir, exist_ok=True)
    os.makedirs(resistance_dir, exist_ok=True)
    os.makedirs(recency_dir, exist_ok=True)
    os.makedirs(combined_dir, exist_ok=True)

    if tf_strength is None:
        tf_strength = {name: 1.0 for name, _ in dfs}

    dfs_sorted = sorted(dfs, key=lambda x: tf_strength.get(x[0], 1), reverse=True)
    base_name, base_df = dfs_sorted[0]

    for idx in tqdm(range(window, len(base_df)), desc="Generating gradient zone maps"):
        win = base_df.iloc[idx - window:idx].reset_index(drop=True)
        margin = 50
        W_pad = int(W * (1 + right_padding))

        sup_rgb = np.zeros((H, W_pad, 3), dtype=np.uint8)
        res_rgb = np.zeros((H, W_pad, 3), dtype=np.uint8)
        rec_rgb = np.zeros((H, W_pad, 3), dtype=np.uint8)
        comb_rgb = np.zeros((H, W_pad, 3), dtype=np.uint8)   # only sup + res

        start_pos = idx - window
        end_pos   = idx - 1

        # y-scaling
        pmin, pmax = float(win["Low"].min()), float(win["High"].max())
        if pmax == pmin:
            pmax += 1e-6
        def y(p): return int(margin + (pmax - p) * (H - 2*margin) / (pmax - pmin))

        # x-scaling
        step = (W_pad - 2*margin) / (window + zone_extend)
        def x_left(pos):   return int(margin + (pos - start_pos) * step)
        def x_right(pos):  return int(margin + (pos - start_pos + 1) * step)
        def x_center(pos): return int(margin + (pos - start_pos) * step + step/2)

        # iterate TFs
        for tf_name, df in dfs_sorted:
            if idx >= len(df):
                continue

            lifecycle = {}
            for j in range(idx - window, idx):
                if j >= len(df): 
                    continue
                for z in _as_list(df.iloc[j].get("ZoneLifecycle", [])):
                    lifecycle[z["id"]] = z

            for z in lifecycle.values():
                if z.get("confirmed_at") is None:
                    continue
                pivot = z.get("pivot_idx")
                if pivot is None or not (start_pos <= pivot <= end_pos):
                    continue

                end_idx  = z.get("end_idx")
                last_pos = end_pos if (z.get("status") == "Active" and end_idx is None) else (end_idx or end_pos)
                x_start  = x_left(pivot)
                x_end    = x_right(last_pos + zone_extend if (z.get("status") == "Active" and end_idx is None) else last_pos)

                zlow, zhigh, ztype = z["low"], z["high"], z["type"]
                if pd.isna(zlow) or pd.isna(zhigh): 
                    continue
                top, bot = sorted((y(zlow), y(zhigh)))

                w = tf_strength.get(tf_name, 1.0)
                base_intensity = int(60 * w)

                if z.get("status") == "Active":
                    if ztype == "Resistance":
                        draw_gradient(res_rgb, x_start, x_end, top, bot, (0,0,255), base_intensity, sigma)
                        draw_gradient(comb_rgb, x_start, x_end, top, bot, (0,0,255), base_intensity, sigma)
                    else:
                        draw_gradient(sup_rgb, x_start, x_end, top, bot, (0,255,0), base_intensity, sigma)
                        draw_gradient(comb_rgb, x_start, x_end, top, bot, (0,255,0), base_intensity, sigma)
                else:
                    # inactive zones ONLY go into recency
                    if ztype == "Resistance":
                        draw_gradient(rec_rgb, x_start, x_end, top, bot, (0,0,255), base_intensity, sigma)
                    else:
                        draw_gradient(rec_rgb, x_start, x_end, top, bot, (0,255,0), base_intensity, sigma)

        # --- Last price marker ---
        last_row   = win.iloc[-1]
        px = x_center(start_pos + len(win) - 1)
        py = int(np.clip(y(last_row.Close), margin+2, H-margin-2))
        radius = max(2, H // 200)

        for canvas in [sup_rgb, res_rgb, rec_rgb, comb_rgb]:
            cv2.circle(canvas, (px, py), radius, (255, 255, 255), -1)

        # --- Save ---
        cv2.imwrite(os.path.join(support_dir, f"zone_{idx}.png"), sup_rgb)
        cv2.imwrite(os.path.join(resistance_dir, f"zone_{idx}.png"), res_rgb)
        cv2.imwrite(os.path.join(recency_dir, f"zone_{idx}.png"), rec_rgb)
        cv2.imwrite(os.path.join(combined_dir, f"zone_{idx}.png"), comb_rgb)

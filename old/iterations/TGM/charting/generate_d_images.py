import os
import cv2
import numpy as np
import pandas as pd
import ast
from tqdm import tqdm


def _as_list(x):
    """Always return a list of dicts from a dataframe cell."""
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return []
    if isinstance(x, list):
        return [z for z in x if isinstance(z, dict)]
    if isinstance(x, str):
        try:
            parsed = ast.literal_eval(x)
            if isinstance(parsed, list):
                return [z for z in parsed if isinstance(z, dict)]
            elif isinstance(parsed, dict):
                return [parsed]
            else:
                return []
        except Exception:
            return []
    if isinstance(x, dict):
        return [x]
    return []


TF_COLORS = {
    "1d": (0, 0, 255),      # Red
    "4h": (255, 0, 255),    # Purple
    "1h": (0, 255, 255),    # Yellow
    "15m": (255, 255, 0),   # Teal
    "5m": (0, 200, 0),      # Green
    "3m": (255, 182, 193),  # Light Pink
    "1m": (255, 255, 255),  # White
}


def get_tf_color(tf_name, fallback=(128, 128, 128)):
    return TF_COLORS.get(tf_name.lower(), fallback)


def ts_to_base_pos(ts, base_df):
    """Map a timestamp from any TF onto the base_df's integer positions."""
    if ts is None:
        return None
    idx = base_df.index.get_indexer([ts], method="nearest")
    if idx.size == 0 or idx[0] == -1:
        return None
    return idx[0]


def generate(
    dfs,
    support_dir="support_out",
    resistance_dir="resistance_out",
    recency_dir="recency_out",
    combined_dir="combined_out",
    candles_dir="candles_out",
    H=800, W=1200,
    tf_strength=None,
    trail_len=100,
    ball_radius=5
):
    os.makedirs(support_dir, exist_ok=True)
    os.makedirs(resistance_dir, exist_ok=True)
    os.makedirs(recency_dir, exist_ok=True)
    os.makedirs(combined_dir, exist_ok=True)
    os.makedirs(candles_dir, exist_ok=True)

    # Default TF strengths (line thickness scaling)
    if tf_strength is None:
        tf_strength = {
            "1m": .5,
            "3m": .8,
            "5m": 1,
            "15m": 2,
            "1h": 3,
            "4h": 4,
            "1d": 5,
        }

    # Pick base df (lowest TF for price reference)
    base_name, base_df = sorted(
        dfs,
        key=lambda x: pd.Timedelta(x[1].index.freq or "1min")
    )[0]

    # Ball anchor (center of screen vertically, price drift horizontally)
    cx, cy = W // 2, H // 2

    # Store past ball positions for trail
    trail = []

    for idx in tqdm(range(1, len(base_df)), desc="Generating Tron-road charts"):
        # Fresh canvases
        sup_rgb = np.ones((H, W, 3), dtype=np.uint8) * 255
        res_rgb = np.ones((H, W, 3), dtype=np.uint8) * 255
        rec_rgb = np.ones((H, W, 3), dtype=np.uint8) * 255
        comb_rgb = np.ones((H, W, 3), dtype=np.uint8) * 255
        candles = np.ones((H, W, 3), dtype=np.uint8) * 255

        # Current price
        price_now = base_df.iloc[idx].Close
        if pd.isna(price_now):
            continue

        # Map price → horizontal offset
        pmin = float(base_df["Low"].iloc[max(0, idx-200):idx].min())
        pmax = float(base_df["High"].iloc[max(0, idx-200):idx].max())
        if pmax == pmin: 
            pmax += 1e-6
        price_offset = (price_now - (pmin+pmax)/2) / (pmax-pmin) * (W//3)

        # Ball position drifts with price
        ball_x = int(cx + price_offset)
        ball_y = cy

        # Add ball position to trail
        trail.append((ball_x, ball_y))
        if len(trail) > trail_len:
            trail.pop(0)

        # --- Draw TF lanes (midpoint line for each active zone) ---
        for tf_name, df in dfs:
            if idx >= len(df): 
                continue
            for z in _as_list(df.iloc[idx].get("ZoneLifecycle", [])):
                if z.get("status") != "Active":
                    continue
                zlow, zhigh = z.get("low"), z.get("high")
                if pd.isna(zlow) or pd.isna(zhigh):
                    continue

                # Midpoint of zone
                mid_price = (zlow + zhigh) / 2
                lane_x = int(cx + (mid_price - (pmin+pmax)/2) / (pmax-pmin) * (W//3))

                color = get_tf_color(tf_name)
                strength = tf_strength.get(tf_name, 1)
                thickness = max(1, int(strength))

                for canvas in [sup_rgb, res_rgb, comb_rgb, candles]:
                    cv2.line(canvas, (lane_x, 0), (lane_x, H), color, thickness)

        # --- Draw trail (gray fading dots, same size as ball) ---
        for t, (tx, ty) in enumerate(reversed(trail[:-1])):  # skip last = ball
            fade = int(255 * (t / len(trail)))
            cv2.circle(comb_rgb, (tx, int(ty + (t+1) * 8)), ball_radius, (fade, fade, fade), -1)
            cv2.circle(candles, (tx, int(ty + (t+1) * 8)), ball_radius, (fade, fade, fade), -1)

        # --- Draw ball (solid red) ---
        cv2.circle(comb_rgb, (ball_x, ball_y), ball_radius, (0, 0, 255), -1)
        cv2.circle(candles, (ball_x, ball_y), ball_radius, (0, 0, 255), -1)

        # Save images
        cv2.imwrite(os.path.join(support_dir, f"zone_{idx}.png"), sup_rgb)
        cv2.imwrite(os.path.join(resistance_dir, f"zone_{idx}.png"), res_rgb)
        cv2.imwrite(os.path.join(recency_dir, f"zone_{idx}.png"), rec_rgb)
        cv2.imwrite(os.path.join(combined_dir, f"zone_{idx}.png"), comb_rgb)
        cv2.imwrite(os.path.join(candles_dir, f"zone_{idx}.png"), candles)

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


def draw_zone(canvas, x_start, x_end, top, bot, color, vertical=True, reverse=False, strength=1.0):
    """Draw rectangle zone with gradient transparency + boundary line for visibility."""
    H, W, _ = canvas.shape
    x_start = max(0, x_start)
    x_end = min(W, x_end)
    top = max(0, top)
    bot = min(H, bot)
    if x_start >= x_end or top >= bot:
        return

    zone_h = bot - top
    zone_w = x_end - x_start

    # create gradient mask
    if vertical:
        alpha_channel = np.linspace(1.0, 0.0, zone_h, dtype=np.float32)
        if reverse:
            alpha_channel = alpha_channel[::-1]
        alpha_mask = alpha_channel.reshape(zone_h, 1)
        alpha_mask = np.repeat(alpha_mask, zone_w, axis=1)
    else:
        alpha_channel = np.linspace(1.0, 0.0, zone_w, dtype=np.float32)
        if reverse:
            alpha_channel = alpha_channel[::-1]
        alpha_mask = alpha_channel.reshape(1, zone_w)
        alpha_mask = np.repeat(alpha_mask, zone_h, axis=0)

    # scale mask by timeframe strength
    alpha_mask *= strength

    # solid color zone
    zone_color = np.zeros((zone_h, zone_w, 3), dtype=np.float32)
    zone_color[:] = color

    # existing patch
    patch = canvas[top:bot, x_start:x_end].astype(np.float32)

    # alpha blend
    blended = (alpha_mask[..., None] * zone_color + (1 - alpha_mask[..., None]) * patch)
    canvas[top:bot, x_start:x_end] = blended.astype(np.uint8)

    # --- Add boundary line (always visible) ---
    thickness = max(1, int(1 + strength * 2))  # 1px for 1m, up to ~3px for daily
    if reverse:
        # support: strong at bottom
        cv2.line(canvas, (x_start, bot - 1), (x_end, bot - 1), color, thickness)
    else:
        # resistance: strong at top
        cv2.line(canvas, (x_start, top), (x_end, top), color, thickness)


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


def catmull_rom_spline(P0, P1, P2, P3, n_points=10):
    """Generate points on a Catmull-Rom spline between P1 and P2."""
    alpha = 0.5
    t0 = 0.0
    t1 = ((np.linalg.norm(np.subtract(P1, P0))) ** alpha) + t0
    t2 = ((np.linalg.norm(np.subtract(P2, P1))) ** alpha) + t1
    t3 = ((np.linalg.norm(np.subtract(P3, P2))) ** alpha) + t2

    def tj(ti, Pi, Pj):
        return ((np.linalg.norm(np.subtract(Pj, Pi))) ** alpha) + ti

    ts = np.linspace(t1, t2, n_points)
    points = []
    for t in ts:
        A1 = (t1 - t) / (t1 - t0) * np.array(P0) + (t - t0) / (t1 - t0) * np.array(P1)
        A2 = (t2 - t) / (t2 - t1) * np.array(P1) + (t - t1) / (t2 - t1) * np.array(P2)
        A3 = (t3 - t) / (t3 - t2) * np.array(P2) + (t - t2) / (t3 - t2) * np.array(P3)
        B1 = (t2 - t) / (t2 - t0) * A1 + (t - t0) / (t2 - t0) * A2
        B2 = (t3 - t) / (t3 - t1) * A2 + (t - t1) / (t3 - t1) * A3
        C = (t2 - t) / (t2 - t1) * B1 + (t - t1) / (t2 - t1) * B2
        points.append(C)
    return points


def smooth_tail(points, n_points=5):
    """Return smoothed points through Catmull-Rom splines."""
    if len(points) < 4:
        return points
    smoothed = []
    for i in range(len(points) - 3):
        P0, P1, P2, P3 = points[i], points[i+1], points[i+2], points[i+3]
        smoothed.extend(catmull_rom_spline(P0, P1, P2, P3, n_points))
    return smoothed


def generate(
    dfs,
    support_dir="support_out",
    resistance_dir="resistance_out",
    recency_dir="recency_out",
    combined_dir="combined_out",
    candles_dir="candles_out",
    window=500,
    zone_extend=20,
    right_padding=0.2,
    H=800, W=1200,
    tf_strength=None
):
    os.makedirs(support_dir, exist_ok=True)
    os.makedirs(resistance_dir, exist_ok=True)
    os.makedirs(recency_dir, exist_ok=True)
    os.makedirs(combined_dir, exist_ok=True)
    os.makedirs(candles_dir, exist_ok=True)

    # Default strengths if not passed in
    if tf_strength is None:
        tf_strength = {
            "1m": 0.1,
            "3m": 0.2,
            "5m": 0.3,
            "15m": 0.5,
            "1h": 0.7,
            "4h": 0.9,
            "1d": 1.0,
        }

    # sort TFs by weight (so higher TFs dominate overlays)
    dfs_sorted = sorted(dfs, key=lambda x: tf_strength.get(x[0], 0.5), reverse=True)

    # pick the *lowest* timeframe df as base candles
    base_name, base_df = sorted(
        dfs,
        key=lambda x: pd.Timedelta(x[1].index.freq or "1min")
    )[0]

    tail_length = 30  # number of candles for the tail

    for idx in tqdm(range(window, len(base_df)), desc="Generating charts"):
        win = base_df.iloc[idx - window:idx].reset_index(drop=True)
        margin = 50
        W_pad = int(W * (1 + right_padding))

        sup_rgb = np.zeros((H, W_pad, 3), dtype=np.uint8)
        res_rgb = np.zeros((H, W_pad, 3), dtype=np.uint8)
        rec_rgb = np.zeros((H, W_pad, 3), dtype=np.uint8)
        comb_rgb = np.zeros((H, W_pad, 3), dtype=np.uint8)
        candles = np.ones((H, W_pad, 3), dtype=np.uint8) * 255

        start_pos = idx - window
        end_pos = idx - 1

        # y-scaling
        pmin, pmax = float(win["Low"].min()), float(win["High"].max())
        if pmax == pmin:
            pmax += 1e-6

        def y(p): return int(margin + (pmax - p) * (H - 2 * margin) / (pmax - pmin))

        # x-scaling
        step = (W_pad - 2 * margin) / (window + zone_extend)

        def x_left(pos): return int(margin + (pos - start_pos) * step)
        def x_right(pos): return int(margin + (pos - start_pos + 1) * step)
        def x_center(pos): return int(margin + (pos - start_pos) * step + step / 2)

        body_w = max(2, int(step * 0.6))

        # --- Draw candlesticks ---
        for j, row in win.iterrows():
            if pd.isna(row.Open) or pd.isna(row.Close):
                continue
            o, c, h, l = row.Open, row.Close, row.High, row.Low
            x = x_center(start_pos + j)
            col = (100, 100, 100) if c >= o else (50, 50, 50)
            cv2.line(candles, (x, y(h)), (x, y(l)), col, 1)
            cv2.rectangle(
                candles,
                (x - body_w // 2, y(max(o, c))),
                (x + body_w // 2, y(min(o, c))),
                col, -1
            )

        # --- Zones from all TFs ---
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

                # Convert pivot to base position
                pivot_idx = z.get("pivot_idx")
                pivot_ts = df.index[pivot_idx] if pivot_idx is not None and pivot_idx < len(df) else None
                pivot_pos = ts_to_base_pos(pivot_ts, base_df)
                if pivot_pos is None or not (start_pos <= pivot_pos <= end_pos):
                    continue

                # Convert end_idx to base position
                end_idx = z.get("end_idx")
                end_ts = df.index[end_idx] if end_idx is not None and end_idx < len(df) else None
                end_pos_tf = ts_to_base_pos(end_ts, base_df) if end_ts is not None else None

                if z.get("status") == "Active" and end_pos_tf is None:
                    last_pos = end_pos
                else:
                    last_pos = end_pos_tf if end_pos_tf is not None else end_pos

                x_start = x_left(pivot_pos)
                x_end = x_right(last_pos + (zone_extend if z.get("status") == "Active" and end_pos_tf is None else 0))

                zlow, zhigh, ztype = z["low"], z["high"], z["type"]
                if pd.isna(zlow) or pd.isna(zhigh):
                    continue
                top, bot = sorted((y(zlow), y(zhigh)))

                color = get_tf_color(tf_name)
                strength = tf_strength.get(tf_name, 0.5)

                if z.get("status") == "Active":
                    if ztype == "Resistance":
                        draw_zone(res_rgb, x_start, x_end, top, bot, color,
                                  vertical=True, reverse=False, strength=strength)
                        draw_zone(comb_rgb, x_start, x_end, top, bot, color,
                                  vertical=True, reverse=False, strength=strength)
                    else:  # Support
                        draw_zone(sup_rgb, x_start, x_end, top, bot, color,
                                  vertical=True, reverse=True, strength=strength)
                        draw_zone(comb_rgb, x_start, x_end, top, bot, color,
                                  vertical=True, reverse=True, strength=strength)

                    # semi-transparent overlay on candles
                    overlay = candles.copy()
                    cv2.rectangle(overlay, (x_start, top), (x_end, bot), color, -1)
                    candles = cv2.addWeighted(overlay, 0.25 * strength, candles, 1 - 0.25 * strength, 0)
                else:
                    # past zones go to recency
                    draw_zone(rec_rgb, x_start, x_end, top, bot, color,
                              vertical=True, reverse=(ztype == "Support"), strength=strength)

        # --- Last price marker with smoothed tail ---
        if idx >= window + tail_length:
            raw_tail = []
            for k in range(tail_length):
                row = base_df.iloc[idx - k - 1]
                if pd.isna(row.Close):
                    continue
                px = x_center(end_pos - k)
                py = int(np.clip(y(row.Close), margin+2, H-margin-2))
                raw_tail.append((px, py))

            raw_tail = raw_tail[::-1]  # oldest → newest
            smoothed = smooth_tail(raw_tail, n_points=8)

            # draw tail with fading alpha
            for i in range(1, len(smoothed)):
                alpha = i / len(smoothed)
                col = (int(255 * (1 - alpha)),) * 3
                for canvas in [sup_rgb, res_rgb, rec_rgb, comb_rgb, candles]:
                    cv2.line(canvas,
                             tuple(map(int, smoothed[i-1])),
                             tuple(map(int, smoothed[i])),
                             col, 1)

        # draw the "ball" itself
        last_row = win.iloc[-1]
        if not pd.isna(last_row.Close):
            px = x_center(start_pos + len(win) - 1)
            py = int(np.clip(y(last_row.Close), margin+2, H-margin-2))
            radius = max(3, H // 200)
            for canvas in [sup_rgb, res_rgb, rec_rgb, comb_rgb, candles]:
                cv2.circle(canvas, (px, py), radius, (255, 255, 255), -1)

        # --- Save ---
        cv2.imwrite(os.path.join(support_dir, f"zone_{idx}.png"), sup_rgb)
        cv2.imwrite(os.path.join(resistance_dir, f"zone_{idx}.png"), res_rgb)
        cv2.imwrite(os.path.join(recency_dir, f"zone_{idx}.png"), rec_rgb)
        cv2.imwrite(os.path.join(combined_dir, f"zone_{idx}.png"), comb_rgb)
        cv2.imwrite(os.path.join(candles_dir, f"zone_{idx}.png"), candles)

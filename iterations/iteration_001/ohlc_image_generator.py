"""
File:           ohlc_image_generator.py
Description:    Simple OHLC image generator (1 image per window, 512x512 output, adaptive sizing)
Author:         Morgan Cooper
Created:        2025-09-01
Updated:        2025-09-09

Notes:

In each image, we normalize the first day closing price to one, and
construct each subsequent daily close from returns (RETt) according to:

    pt+1 = (1 + RETt+1) * pt
    
Added lagged feature for raw data baseline modeling.

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from time import time
from tqdm import tqdm


def ohlc_image_generator(
    ohlc_df: pd.DataFrame = pd.DataFrame(),
    open_col: str = "Open",
    high_col: str = "High",
    low_col: str = "Low",
    close_col: str = "Close",
    window: int = 180,
    pred_window: int = 60,
    save_dir: str | None = None,
    verbose: bool = True,
    y_pad_pct: float = 0.08,               # vertical padding as % of range
    ref_window_for_scale: int = 60,        # reference window used to scale linewidths/arms
    px_per_bar: int = 3,                   # width in pixels allocated per OHLC bar
    rebuild: bool = False,                 # if False, skip items already listed in meta.csv
    non_overlap_after_horizon: bool = True # if True, step by (window + pred_window)
):

    start_time = time()

    # Image output config with 3px per bar (+1 allows clean padding at image edges)
    img_px = (window * px_per_bar) + 1
    dpi = 512
    figsize_inch = img_px / dpi

    # Save path setup
    save_dir = Path(save_dir) if save_dir else Path(f"../../data/ohlc_images/window={window}")
    save_dir.mkdir(parents=True, exist_ok=True)
    meta_out = save_dir / "meta.csv"

    # If not rebuilding, load existing meta and collect existing filenames to skip
    existing_names: set[str] = set()
    if not rebuild and meta_out.exists():
        try:
            _meta = pd.read_csv(meta_out, usecols=["path"])
            existing_names = set(Path(str(p)).name for p in _meta["path"].astype(str).tolist())
        except Exception as e:
            if verbose:
                print(f"[warn] failed reading existing meta ({e}); proceeding without skip list)")

    df = ohlc_df.copy().reset_index(drop=True)
    total_rows = len(df)
    min_required = window + pred_window
    if total_rows < min_required:
        raise ValueError(f"Need at least {min_required} rows, got {total_rows}")

    # Step by (window + pred_window) to ensure a gap equal to horizon
    step = (pred_window) if non_overlap_after_horizon else 1
    iterator = tqdm(
        range(min_required, total_rows, step),
        desc="Generating OHLC images",
        disable=not verbose
    )

    # helper to convert pixel widths to matplotlib points (linewidth units)
    def px_to_pts(px: float) -> float:
        return px * 72.0 / dpi

    # Scale drawing params so visuals look consistent as window changes
    scale = (ref_window_for_scale / max(window, 1)) ** 0.5
    one_px = px_to_pts(1.0)

    for i in iterator:
        image_start = i - pred_window - window
        image_end   = i - pred_window      # slice end (exclusive)
        future_idx  = i                    # label index (exactly pred_window after image_end-1)

        if image_start < 0 or future_idx >= total_rows:
            continue

        fname = f"ohlc_{image_start}_{image_end}.png"

        # Skip if already listed in meta and rebuild=False
        if not rebuild and fname in existing_names:
            if verbose:
                print(f"[skip-existing] {fname}")
            continue

        window_df = df.iloc[image_start:image_end].copy()

        # Compute per-window arithmetic returns from Close, then normalized Close via cumulative product
        # This yields the same effect as C_t / C_0 within the window, but keeps the "returns -> price" logic explicit.
        close_vals = window_df[close_col].astype(float).values
        r = np.zeros_like(close_vals, dtype=float)
        if len(close_vals) > 1:
            prev = close_vals[:-1]
            curr = close_vals[1:]
            with np.errstate(divide='ignore', invalid='ignore'):
                r[1:] = np.where(prev != 0.0, (curr / prev) - 1.0, 0.0)
        norm_close = np.cumprod(1.0 + r)

        # Scale O/H/L in proportion to that day’s closing price level
        with np.errstate(divide='ignore', invalid='ignore'):
            rc = close_vals
            safe_rc = np.where(rc == 0.0, 1.0, rc)
            ratio_o = window_df[open_col].astype(float).values / safe_rc
            ratio_h = window_df[high_col].astype(float).values / safe_rc
            ratio_l = window_df[low_col].astype(float).values  / safe_rc

        o_hat = ratio_o * norm_close
        h_hat = ratio_h * norm_close
        l_hat = ratio_l * norm_close
        c_hat = norm_close  # by construction

        # Compute vertical limits with consistent padding
        ymin = float(np.min(l_hat))
        ymax = float(np.max(h_hat))
        yrange = ymax - ymin if ymax > ymin else max(1e-9, abs(ymax) * 1e-6)
        pad = yrange * y_pad_pct
        ylo, yhi = ymin - pad, ymax + pad

        fig, ax = plt.subplots(figsize=(figsize_inch, figsize_inch), dpi=dpi)
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

        # Horizontal X pixel location per bar: center of its allocated px_per_bar slot
        x_idx = np.array([j * px_per_bar + px_per_bar // 2 for j in range(window)])

        for j in range(window):
            o, h, l, c = o_hat[j], h_hat[j], l_hat[j], c_hat[j]
            x = x_idx[j]
            color = "green" if c >= o else "red"

            # Wick (Low -> High)
            ax.plot([x, x], [l, h],
                    color=color,
                    linewidth=one_px,
                    antialiased=False,
                    solid_capstyle='butt')

            # Arms (Open and Close)
            arm_len = (px_per_bar / 2) - 0.5
            ax.plot([x - arm_len, x], [o, o],
                    color=color, linewidth=one_px,
                    antialiased=False, solid_capstyle='butt')
            ax.plot([x, x + arm_len], [c, c],
                    color=color, linewidth=one_px,
                    antialiased=False, solid_capstyle='butt')

        half_bar = px_per_bar / 2
        ax.set_xlim(-half_bar, img_px - 1 + half_bar)
        ax.set_ylim(ylo, yhi)

        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Save image
        save_path = save_dir / fname
        fig.savefig(save_path, dpi=dpi, facecolor="white", bbox_inches=None, pad_inches=0)
        plt.close(fig)

        # --- Meta with lagged features ---
        # Baseline model use on raw features
        c_now = float(df.loc[image_end - 1, close_col])
        c_fut = float(df.loc[future_idx,    close_col])
        label = 1 if c_fut > c_now else 0

        lag_features = {}
        num_lags = window
        lag_cols = [open_col, high_col, low_col, close_col, "Volume"]

        for col in lag_cols:
            for l in range(1, num_lags + 1):
                val = df.loc[image_end - l, col]
                lag_features[f"{col}_lag{l}"] = val

        row = {"path": str(save_path), "label": label}
        row.update(lag_features)

        # Append to meta.csv (header only if file doesn't exist yet)
        try:
            pd.DataFrame([row]).to_csv(meta_out, mode="a", header=not meta_out.exists(), index=False)
        except Exception as e:
            try:
                save_path.unlink(missing_ok=True)
            except Exception:
                pass
            if verbose:
                print(f"[WARN] Meta write failed for {save_path}: {e}. Image removed.")

    if verbose:
        print(f"Completed in {round(time() - start_time, 2)}s")

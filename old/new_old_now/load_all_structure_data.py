import os
import pandas as pd

def load_all_structure_data(
    resampled_dir="resampled_data",
    level_dir="break_level_data",
    event_dir="break_event_data"
):
    candle_data = {}
    break_levels = {}
    break_events = {}

    # --- Helper: extract TF from filenames ---
    def extract_tf(file, prefix):
        return file.replace(prefix, "").replace(".csv", "")

    # --- Discover all available timeframes ---
    tf_candles = {
        extract_tf(f, "ohlcv_")
        for f in os.listdir(resampled_dir)
        if f.startswith("ohlcv_") and f.endswith(".csv")
    }

    tf_levels = {
        extract_tf(f, "break_levels_")
        for f in os.listdir(level_dir)
        if f.startswith("break_levels_") and f.endswith(".csv")
    }

    tf_events = {
        extract_tf(f, "break_events_")
        for f in os.listdir(event_dir)
        if f.startswith("break_events_") and f.endswith(".csv")
    }

    # --- Get only consistent timeframes across all three ---
    valid_tfs = tf_candles & tf_levels & tf_events

    if not valid_tfs:
        raise ValueError("❌ No common timeframes found across all data sources.")

    # --- Warn about partial data ---
    all_tfs = tf_candles | tf_levels | tf_events
    missing_tfs = all_tfs - valid_tfs
    if missing_tfs:
        print(f"⚠️ Ignoring incomplete timeframes (not present in all 3 dirs): {sorted(missing_tfs)}")

    # --- Load data only for valid timeframes ---
    for tf in sorted(valid_tfs):
        candle_path = os.path.join(resampled_dir, f"ohlcv_{tf}.csv")
        level_path = os.path.join(level_dir, f"break_levels_{tf}.csv")
        event_path = os.path.join(event_dir, f"break_events_{tf}.csv")

        candle_data[tf] = pd.read_csv(candle_path, index_col=0, parse_dates=True)
        break_levels[tf] = pd.read_csv(level_path, parse_dates=["level_time"])
        break_events[tf] = pd.read_csv(event_path, parse_dates=["break_time"])

    print(f"✅ Loaded timeframes: {sorted(valid_tfs)}")

    return candle_data, break_levels, break_events

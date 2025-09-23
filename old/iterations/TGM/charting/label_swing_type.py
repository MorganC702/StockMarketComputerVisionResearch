# ------------------------------------------------
# Swing Labeling (backfill version)
# ------------------------------------------------
def label_swing_type(open_series, close_series):
    dirs = [1 if c - o > 0 else -1 if c - o < 0 else 0 for o, c in zip(open_series, close_series)]
    n = len(dirs)
    swings = [None] * n
    current_swing = None

    prev_dir = None
    start_streak = 0
    opposite_streak = 0
    backfill_start = None

    for i, d in enumerate(dirs):
        if d == 0:
            swings[i] = current_swing
            continue

        if current_swing is None:
            if prev_dir is None or prev_dir != d:
                prev_dir = d
                start_streak = 1
                swings[i] = None
            else:
                start_streak += 1
                if start_streak >= 2:
                    current_swing = "Accumulation" if d == 1 else "Distribution"
                swings[i] = current_swing
            continue

        same_dir = ((current_swing == "Accumulation" and d == 1) or
                    (current_swing == "Distribution" and d == -1))

        if same_dir:
            opposite_streak = 0
            backfill_start = None
            swings[i] = current_swing
        else:
            if opposite_streak == 0:
                opposite_streak = 1
                backfill_start = i
                swings[i] = current_swing
            else:
                new_swing = "Distribution" if current_swing == "Accumulation" else "Accumulation"
                for j in range(backfill_start, i + 1):
                    swings[j] = new_swing
                current_swing = new_swing
                opposite_streak = 0
                backfill_start = None
                swings[i] = current_swing
    return swings

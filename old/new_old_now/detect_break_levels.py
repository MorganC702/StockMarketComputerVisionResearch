import pandas as pd

def detect_break_levels(df):
    directions = df["Close"].sub(df["Open"]).apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0)
    swing_levels = []

    i = 0
    n = len(df)
    current_swing = None
    extreme_price = None
    extreme_time = None
    opposite_streak = 0

    while i < n:
        d = directions.iloc[i]
        row = df.iloc[i]
        time = df.index[i]

        if d == 0:
            i += 1
            continue

        if current_swing is None:
            current_swing = 'high_break' if d == 1 else 'low_break'
            extreme_price = row["High"] if d == 1 else row["Low"]
            extreme_time = time
            i += 1
            continue

        if ((current_swing == 'high_break' and d == 1) or
            (current_swing == 'low_break' and d == -1)):
            # Extend current swing, update extreme
            if current_swing == 'high_break' and row["High"] > extreme_price:
                extreme_price = row["High"]
                extreme_time = time
            elif current_swing == 'low_break' and row["Low"] < extreme_price:
                extreme_price = row["Low"]
                extreme_time = time
            opposite_streak = 0
        else:
            opposite_streak += 1
            if opposite_streak >= 2:
                # Confirm swing and store level
                swing_levels.append({
                    "type": current_swing,
                    "level_price": extreme_price,
                    "level_time": extreme_time
                })
                # Start new swing
                current_swing = 'low_break' if current_swing == 'high_break' else 'high_break'
                extreme_price = row["High"] if current_swing == 'high_break' else row["Low"]
                extreme_time = time
                opposite_streak = 0
        i += 1

    return pd.DataFrame(swing_levels)


def track_break_events(df, levels_df):
    df = df.sort_index()
    levels_df = levels_df.sort_values("level_time")

    high_stack = []
    low_stack = []
    break_events = []

    # Preload stacks with all levels
    for _, row in levels_df.iterrows():
        entry = {
            "level_price": row["level_price"],
            "level_time": row["level_time"],
            "type": row["type"]
        }
        if row["type"] == "high_break":
            high_stack.append(entry)
        else:
            low_stack.append(entry)

    # Sort stacks by time (just in case)
    high_stack.sort(key=lambda x: x["level_time"])
    low_stack.sort(key=lambda x: x["level_time"])

    for timestamp, row in df.iterrows():
        close = row["Close"]

        # Use regular loop to safely modify list
        new_high_stack = []
        for lvl in high_stack:
            if lvl["level_time"] <= timestamp and close > lvl["level_price"]:
                break_events.append({
                    "break_time": timestamp,
                    "level_price": lvl["level_price"],
                    "break_type": "high_break"
                })
            else:
                new_high_stack.append(lvl)
        high_stack = new_high_stack

        new_low_stack = []
        for lvl in low_stack:
            if lvl["level_time"] <= timestamp and close < lvl["level_price"]:
                break_events.append({
                    "break_time": timestamp,
                    "level_price": lvl["level_price"],
                    "break_type": "low_break"
                })
            else:
                new_low_stack.append(lvl)
        low_stack = new_low_stack

    return pd.DataFrame(break_events)

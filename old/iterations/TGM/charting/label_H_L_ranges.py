# ------------------------------------------------
# Running Highs / Lows of Swings
# ------------------------------------------------
def label_ranges(high_series, low_series, swing_series):
    """
    Track both:
      - Running high of the most recent Accumulation swing
      - Running low of the most recent Distribution swing
    Always save both values each step.
    """
    range_high, range_low = [], []
    current_high, current_low = None, None
    current_swing = None

    for high, low, swing in zip(high_series, low_series, swing_series):
        # Swing change -> reset appropriate tracker
        if swing != current_swing:
            current_swing = swing
            if swing == "Accumulation":
                current_high = high
            elif swing == "Distribution":
                current_low = low

        # Update running trackers
        if swing == "Accumulation":
            current_high = high if current_high is None else max(current_high, high)
        elif swing == "Distribution":
            current_low = low if current_low is None else min(current_low, low)

        # Save both at each row
        range_high.append(current_high)
        range_low.append(current_low)

    return range_high, range_low

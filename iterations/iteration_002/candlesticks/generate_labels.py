import pandas as pd

def label(open_val, high_val, low_val, close_val) -> str:
    body_size = abs(open_val - close_val)
    high_low_range = high_val - low_val
    if high_low_range == 0:
        return "Doji"
    
    body_ratio = body_size / high_low_range
    upper_wick_ratio = (high_val - max(close_val, open_val)) / high_low_range
    lower_wick_ratio = (min(close_val, open_val) - low_val) / high_low_range

    # Bearish candles
    if close_val < open_val:
        if body_ratio >= 0.7 and upper_wick_ratio <= 0.15 and lower_wick_ratio <= 0.15:
            return "High Bearish Sentiment"
        elif body_ratio >= 0.55 and upper_wick_ratio <= 0.2 and lower_wick_ratio <= 0.2:
            return "Medium-High Bearish Sentiment"
        elif body_ratio >= 0.4:
            return "Neutral Bearish Sentiment"
        elif body_ratio >= 0.25:
            return "Medium-Low Bearish Sentiment"
        else:
            return "Low Bearish Sentiment"
    
    # Bullish candles
    elif close_val > open_val:
        if body_ratio >= 0.7 and upper_wick_ratio <= 0.15 and lower_wick_ratio <= 0.15:
            return "High Bullish Sentiment"
        elif body_ratio >= 0.55 and upper_wick_ratio <= 0.2 and lower_wick_ratio <= 0.2:
            return "Medium-High Bullish Sentiment"
        elif body_ratio >= 0.4:
            return "Neutral Bullish Sentiment"
        elif body_ratio >= 0.25:
            return "Medium-Low Bullish Sentiment"
        else:
            return "Low Bullish Sentiment"
    
    # Doji / Flat
    else:
        return "Doji"

def generate_labels(
    df: pd.DataFrame,
    open_col: str = "Open",
    high_col: str = "High",
    low_col: str = "Low",
    close_col: str = "Close"
) -> pd.DataFrame: 
    
    df = df.copy()
    
    # Apply the labeling function row by row
    df["Label"] = df.apply(
        lambda row: label(
            open_val=row[open_col],
            high_val=row[high_col],
            low_val=row[low_col],
            close_val=row[close_col]
        ), axis=1
    )
    
    return df

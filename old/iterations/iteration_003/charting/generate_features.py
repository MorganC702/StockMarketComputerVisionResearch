from charting.label_candle_type import label_candle_type
from charting.label_swing_type import label_swing_type
from charting.label_H_L_ranges import label_ranges
from charting.label_reversal_zones import label_unconfirmed_zones, track_zone_lifecycle
import pandas as pd
import numpy as np 

# ------------------------------------------------
# 6. Main Feature Generator
# ------------------------------------------------

def generate_features(
    df: pd.DataFrame,
    open_col: str = "Open",
    high_col: str = "High",
    low_col: str = "Low",
    close_col: str = "Close",
) -> pd.DataFrame:
    df = df.copy()

    df["CandleDirectionLabel"] = df.apply(
        lambda row: label_candle_type(
            open_val=row[open_col],
            high_val=row[high_col],
            low_val=row[low_col],
            close_val=row[close_col]
        ), axis=1
    )

    df["SwingTypeLabel"] = label_swing_type(df[open_col].values, df[close_col].values)
    
    df["RangeHigh"], df["RangeLow"] = label_ranges(df[high_col].values, df[low_col].values, df["SwingTypeLabel"].values)


    zdf = label_unconfirmed_zones(df, high_col=high_col, low_col=low_col, swing_col="SwingTypeLabel")
    cols_to_join = ["ZoneLow","ZoneHigh","ZoneType","ZoneId","ZonePivotIdx"]
    df[cols_to_join] = zdf[cols_to_join]

    df = track_zone_lifecycle(
        df,
        zone_low_col="ZoneLow", 
        zone_high_col="ZoneHigh",
        zone_type_col="ZoneType", 
        zone_id_col="ZoneId",
        pivot_idx_col="ZonePivotIdx"
    )
    return df

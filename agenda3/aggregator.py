import pandas as pd
from collections import deque

class TimeframeAggregator:
    def __init__(self, buffer: deque):
        self.buffer = buffer

    def resample_all(self, timeframes: list[str]) -> dict:
        df = pd.DataFrame(list(self.buffer)) 
        df.set_index("timestamp", inplace=True)

        resampled = {}
        for tf in timeframes:
            rule = self._get_pandas_rule(tf)
            grouped = df.resample(rule, label="right", closed="right").agg({
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
            }).dropna()
            grouped.reset_index(inplace=True)
            resampled[tf] = grouped

        return resampled

    def _get_pandas_rule(self, tf: str) -> str:
        return {
            "5m" :  "5min",
            "15m":  "15min",
            "1h" :  "1h",
            "4h" :  "4h",
            "1d" :  "1d",
        }[tf]

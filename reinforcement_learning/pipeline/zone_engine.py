from collections import deque
from zone_logic.zone_tracker import ZoneTracker
import pandas as pd 


class ZoneEngine:
    def __init__(self, timeframes: list[str]):
        self.trackers = {}
        for tf in timeframes:
            self.trackers[tf] = ZoneTracker(
                pending_support_stack=deque(),
                pending_resistance_stack=deque(),
                active_support_stack=deque(),
                active_resistance_stack=deque(),
                deactivated_support_stack=deque(),
                deactivated_resistance_stack=deque(),
            )
    
    
    def _parse_tf_to_minutes(self, tf: str) -> int:
        mapping = {
            "1m": 1, 
            "3m": 3, 
            "5m": 5,
            "15m": 15, 
            "1h": 60,
            "4h": 240, 
            "1d": 1440
        }
        return mapping[tf]


    def update(self, tf: str, bar: dict):
        tracker = self.trackers[tf]

        ts = pd.to_datetime(bar["timestamp"])
        tf_minutes = self._parse_tf_to_minutes(tf)
        aligned_ts = ts.floor(f"{tf_minutes}min")

        if ts != aligned_ts:
            return 

        tracker.update(bar)


    def get_visible_zones(self, tf: str, linger_bars: int = 60):
        return self.trackers[tf].get_visible_zones(linger_bars=linger_bars)

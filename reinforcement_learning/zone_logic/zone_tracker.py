from collections import deque
from datetime import timedelta

class ZoneTracker:
    def __init__(
        self,
        pending_support_stack: deque,
        pending_resistance_stack: deque,
        active_support_stack: deque,
        active_resistance_stack: deque,
        deactivated_support_stack: deque,
        deactivated_resistance_stack: deque,
 
    ):
        
        CANDLE_BUFFER_SIZE = 180 
        self.candle_buffer = deque(maxlen=CANDLE_BUFFER_SIZE)
        
        self.directional_streak = []
    
        # Zone stacks for tracking zones
        self.active_support_stack = active_support_stack
        self.active_resistance_stack = active_resistance_stack
        self.pending_support_stack = pending_support_stack
        self.pending_resistance_stack = pending_resistance_stack
        self.deactivated_support_stack = deactivated_support_stack
        self.deactivated_resistance_stack = deactivated_resistance_stack


    def reset(self):
        self.candle_buffer.clear()
        self.directional_streak.clear()
        self.active_support_stack.clear()
        self.active_resistance_stack.clear()
        self.pending_support_stack.clear()
        self.pending_resistance_stack.clear()
        self.deactivated_support_stack.clear()
        self.deactivated_resistance_stack.clear()


    def update(self, bar):
        self.candle_buffer.append(bar)

        # 1 == up candle, -1 == down candle, 0 == flat candle
        o, c = bar["open"], bar["close"]
        candle_direction = 1 if c > o else -1 if c < o else 0

        # Track directional streak
        # Skip flat (neutral sentiment) candles when calculating swings
        if candle_direction in (-1, 1):
            self.directional_streak.append(candle_direction)
            if len(self.directional_streak) > 24:
                self.directional_streak.pop(0)

        # Detect Reversal
        filtered_candle_history = [d for d in self.directional_streak if d != 0] # remove flat candles for calulating pivots
        if len(filtered_candle_history) >= 4:
            last4 = filtered_candle_history[-4:]
            
            if last4[:2] == [1, 1] and last4[2:] == [-1, -1]:
                self._add_to_pending_zones_stack(reversal_type=1) # 1 = Peak (up up down down) 
            elif last4[:2] == [-1, -1] and last4[2:] == [1, 1]:
                self._add_to_pending_zones_stack(reversal_type=-1) # -1 = Valley (down down up up) 


        # Check for invaldations or activations on current pending zones
        self._check_pending_zones()

        self._check_active_zones()


    def _add_to_pending_zones_stack(self, reversal_type):
        
        if len(self.candle_buffer) < 4:
            return

        recent_bars = list(self.candle_buffer)[-4:]
        second_candle = recent_bars[1]

        if reversal_type == 1:
            zone_high = max(bar["high"] for bar in recent_bars)
            zone_low = second_candle["low"]
        else:
            zone_high = second_candle["high"]
            zone_low = min(bar["low"] for bar in recent_bars)

        zone = {
            "high": zone_high,
            "low": zone_low,
            "zone_type": reversal_type, 
            "pivot_timestamp": second_candle["timestamp"],
            "pending_placement_timestamp": recent_bars[-1]["timestamp"],
            "confirmation_timestamp": None,
            "confirmed": False, 
            "invalidation_timestamp": None,
            "invalidated": False,
            "break_timestamp": None,
            "break_interaction_type": None,
            "zone_broken": False,
        }

        (self.pending_resistance_stack if reversal_type == 1 else self.pending_support_stack).append(zone)



    def _check_pending_zones(self):
        current_bar = self.candle_buffer[-1]
        o, c = current_bar["open"], current_bar["close"]
        candle_direction = 1 if c > o else -1 if c < o else 0
        
        for stack in [self.pending_support_stack, self.pending_resistance_stack]:
            zones_to_remove = []

            for zone in stack:
                invalidated = False

                if zone["zone_type"] == 1:  # Resistance 
                    if current_bar["high"] > zone["high"] or candle_direction == 1:
                        invalidated = True

                else:  # Support 
                    if current_bar["low"] < zone["low"] or candle_direction == -1: 
                        invalidated = True


                if invalidated:
                    zone["invalidated"] = True
                    zone["invalidation_timestamp"] = current_bar["timestamp"]
                    zones_to_remove.append(zone)
                    if zone["zone_type"] == 1:
                        self.deactivated_resistance_stack.append(zone)
                    else:
                        self.deactivated_support_stack.append(zone)

                else: 
                    if zone["zone_type"] == 1:  # Resistance zone confirms on close < low
                        if current_bar["close"] < zone["low"]:
                            zone["confirmed"] = True
                            zone["confirmation_timestamp"] = current_bar["timestamp"]
                            zones_to_remove.append(zone)
                            self.active_resistance_stack.append(zone)
                   
                    else:  # Support zone confirms on close > high
                        if current_bar["close"] > zone["high"]:
                            zone["confirmed"] = True
                            zone["confirmation_timestamp"] = current_bar["timestamp"]
                            zones_to_remove.append(zone)
                            self.active_support_stack.append(zone)

            # Remove all moved zones from pending stack
            for zone in zones_to_remove:
                stack.remove(zone)


    def _check_active_zones(self):
        current_bar = self.candle_buffer[-1]
        o, c = current_bar["open"], current_bar["close"]
        candle_direction = 1 if c > o else -1 if c < o else 0
        
        for stack in [self.active_support_stack, self.active_resistance_stack]:
            zones_to_remove = []
            for zone in stack:

                # Avoid breaking on same bar as confirmation
                if current_bar["timestamp"] == zone["confirmation_timestamp"]:
                    continue
                    
                if zone["zone_type"] == 1:  # Resistance
                    if current_bar["high"] >= zone["high"]:
                        zone["break_interaction_type"] = "high_break_level_event"
                        zone["break_timestamp"] = current_bar["timestamp"]
                        
                        zone["zone_broken"] = True
                        zones_to_remove.append(zone)
                        self.deactivated_resistance_stack.append(zone)
                    elif (
                        zone["low"] <= current_bar["high"] < zone["high"]
                        and current_bar["close"] < zone["low"]
                        and candle_direction == 1
                    ):
                        zone["break_interaction_type"] = "resistance_zone_wick_event"
                        zone["break_timestamp"] = current_bar["timestamp"]
                        
                        zone["zone_broken"] = True
                        zones_to_remove.append(zone)
                        self.deactivated_resistance_stack.append(zone)
                    elif (
                        zone["low"] <= current_bar["open"] <= zone["high"]
                        and current_bar["close"] < zone["low"] 


                    ):
                        zone["break_interaction_type"] = "resistance_zone_body_event"
                        zone["break_timestamp"] = current_bar["timestamp"]
                        zone["zone_broken"] = True
                        zones_to_remove.append(zone)
                        self.deactivated_resistance_stack.append(zone)
                
                else:  # Support
                    if current_bar["low"] <= zone["low"]:
                        zone["break_interaction_type"] = "support_break_level_event"
                        zone["break_timestamp"] = current_bar["timestamp"]
                        zone["zone_broken"] = True
                        zones_to_remove.append(zone)
                        self.deactivated_support_stack.append(zone)
                    elif (
                        zone["high"] >= current_bar["low"] > zone["low"]
                        and current_bar["close"] > zone["high"]
                        and candle_direction == -1
                    ):
                        zone["break_interaction_type"] = "support_zone_wick_event"
                        zone["break_timestamp"] = current_bar["timestamp"]
                        zone["zone_broken"] = True
                        zones_to_remove.append(zone)
                        self.deactivated_support_stack.append(zone)
                    elif (
                        zone["high"] >= current_bar["open"] >= zone["low"]
                        and current_bar["close"] > zone["high"]
                    ):
                        zone["break_interaction_type"] = "support_zone_body_event"
                        zone["break_timestamp"] = current_bar["timestamp"]
                        zone["zone_broken"] = True
                        zones_to_remove.append(zone)
                        self.deactivated_support_stack.append(zone)
                        
            # Remove all moved zones from active stack
            for zone in zones_to_remove:
                stack.remove(zone)

    def get_visible_zones(self, linger_bars: int = 100):
        visible = list(self.active_support_stack) + list(self.active_resistance_stack)

        if not self.candle_buffer:
            return visible

        current_idx = len(self.candle_buffer) - 1
        current_ts = self.candle_buffer[current_idx]["timestamp"]

        # Get the oldest timestamp still allowed to be visible
        history = list(self.candle_buffer)
        oldest_idx = max(0, current_idx - linger_bars)
        threshold_ts = history[oldest_idx]["timestamp"]

        def is_recent(zone):
            return (
                zone.get("zone_broken") is True
                and zone.get("break_timestamp") is not None
                and zone["break_timestamp"] >= threshold_ts
            )

        lingering_support_zones = [z for z in self.deactivated_support_stack if is_recent(z)]
        lingering_resistance_zones = [z for z in self.deactivated_resistance_stack if is_recent(z)]

        visible += lingering_support_zones + lingering_resistance_zones
        return visible


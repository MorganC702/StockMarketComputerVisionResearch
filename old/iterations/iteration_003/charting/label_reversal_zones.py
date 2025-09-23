import pandas as pd
import numpy as np


# ------------------------------------------------
# Build Unconfirmed Zones (Reversal + Running Extremes)
# ------------------------------------------------
# def label_unconfirmed_zones(
#     df: pd.DataFrame,
#     high_col: str = "High",
#     low_col: str = "Low",
#     swing_col: str = "SwingTypeLabel",
#     lookahead: int = 2,   # how many bars into the new swing to include
# ) -> pd.DataFrame:
#     """
#     Build pivot-stamped support/resistance zones.

#     Rules:
#     - Support (Distribution → Accumulation):
#         ZoneLow  = min(low) across Distribution leg + first N Accum bars
#         ZoneHigh = pivot bar high
#     - Resistance (Accumulation → Distribution):
#         ZoneLow  = pivot bar low
#         ZoneHigh = max(high) across Accum leg + first N Dist bars
#     """

#     out = df.copy()
#     n = len(out)

#     highs  = out[high_col].to_numpy(float)
#     lows   = out[low_col].to_numpy(float)
#     swings = out[swing_col].to_numpy(object)

#     zone_low    = [None] * n
#     zone_high   = [None] * n
#     zone_type   = [None] * n
#     zone_id     = [None] * n
#     zone_pivot  = [None] * n

#     s_count, r_count = 0, 0
#     cur = None
#     seg_start = None

#     for i in range(n):
#         s = swings[i]

#         if s != cur:
#             if cur in ("Accumulation", "Distribution") and seg_start is not None:
#                 seg_end = i - 1
#                 if seg_end >= seg_start:
#                     seg = slice(seg_start, seg_end + 1)
#                     next_seg = slice(i, min(n, i + lookahead))

#                     if cur == "Distribution" and s == "Accumulation":
#                         # ----- Support -----
#                         p = seg_end  # pivot = last bar of distribution
#                         s_count += 1
#                         zid = f"S{s_count}"

#                         vlow = float(np.min(np.r_[lows[seg], lows[next_seg]]))
#                         vhigh = float(highs[p])  # pivot bar high

#                         # safety swap
#                         lo, hi = sorted([vlow, vhigh])

#                         zone_low[p]   = lo
#                         zone_high[p]  = hi
#                         zone_type[p]  = "Support"
#                         zone_id[p]    = zid
#                         zone_pivot[p] = p

#                     elif cur == "Accumulation" and s == "Distribution":
#                         # ----- Resistance -----
#                         p = seg_end  # pivot = last bar of accumulation
#                         r_count += 1
#                         zid = f"R{r_count}"

#                         vlow = float(lows[p])  # pivot bar low
#                         vhigh = float(np.max(np.r_[highs[seg], highs[next_seg]]))

#                         # safety swap
#                         lo, hi = sorted([vlow, vhigh])

#                         zone_low[p]   = lo
#                         zone_high[p]  = hi
#                         zone_type[p]  = "Resistance"
#                         zone_id[p]    = zid
#                         zone_pivot[p] = p

#             # start new segment
#             if s in ("Accumulation", "Distribution"):
#                 cur, seg_start = s, i
#             else:
#                 cur, seg_start = None, None

#     out["ZoneLow"]      = zone_low
#     out["ZoneHigh"]     = zone_high
#     out["ZoneType"]     = zone_type
#     out["ZoneId"]       = zone_id
#     out["ZonePivotIdx"] = zone_pivot
#     return out


import pandas as pd
import numpy as np


# ------------------------------------------------
# Build Unconfirmed Zones (Reversal + Running Extremes)
# ------------------------------------------------
def label_unconfirmed_zones(
    df: pd.DataFrame,
    high_col: str = "High",
    low_col: str = "Low",
    open_col: str = "Open",
    close_col: str = "Close",
    swing_col: str = "SwingTypeLabel",
    lookahead: int = 2,   # how many bars into the new swing to include
) -> pd.DataFrame:
    """
    Build pivot-stamped support/resistance zones.

    Rules:
    - Support (Distribution → Accumulation):
        ZoneLow  = min(low) across Distribution leg + first N Accum bars
        ZoneHigh = pivot bar high
    - Resistance (Accumulation → Distribution):
        ZoneLow  = pivot bar low
        ZoneHigh = max(high) across Accum leg + first N Dist bars

    Extra:
    - 1-bar reversal override:
      * Any bearish bar after Accumulation flips swing to Distribution
      * Any bullish bar after Distribution flips swing to Accumulation
    """

    out = df.copy()
    n = len(out)

    highs  = out[high_col].to_numpy(float)
    lows   = out[low_col].to_numpy(float)
    opens  = out[open_col].to_numpy(float)
    closes = out[close_col].to_numpy(float)
    swings = out[swing_col].to_numpy(object)

    zone_low    = [None] * n
    zone_high   = [None] * n
    zone_type   = [None] * n
    zone_id     = [None] * n
    zone_pivot  = [None] * n

    s_count, r_count = 0, 0
    cur = None
    seg_start = None

    for i in range(n):
        s = swings[i]

        # --- 1-bar reversal override ---
        if cur == "Accumulation" and closes[i] < opens[i]:
            s = "Distribution"
        elif cur == "Distribution" and closes[i] > opens[i]:
            s = "Accumulation"

        if s != cur:
            if cur in ("Accumulation", "Distribution") and seg_start is not None:
                seg_end = i - 1
                if seg_end >= seg_start:
                    seg = slice(seg_start, seg_end + 1)
                    next_seg = slice(i, min(n, i + lookahead))

                    if cur == "Distribution" and s == "Accumulation":
                        # ----- Support -----
                        p = seg_end  # pivot = last bar of distribution
                        s_count += 1
                        zid = f"S{s_count}"

                        vlow = float(np.min(np.r_[lows[seg], lows[next_seg]]))
                        vhigh = float(highs[p])  # pivot bar high

                        # safety swap
                        lo, hi = sorted([vlow, vhigh])

                        zone_low[p]   = lo
                        zone_high[p]  = hi
                        zone_type[p]  = "Support"
                        zone_id[p]    = zid
                        zone_pivot[p] = p

                    elif cur == "Accumulation" and s == "Distribution":
                        # ----- Resistance -----
                        p = seg_end  # pivot = last bar of accumulation
                        r_count += 1
                        zid = f"R{r_count}"

                        vlow = float(lows[p])  # pivot bar low
                        vhigh = float(np.max(np.r_[highs[seg], highs[next_seg]]))

                        # safety swap
                        lo, hi = sorted([vlow, vhigh])

                        zone_low[p]   = lo
                        zone_high[p]  = hi
                        zone_type[p]  = "Resistance"
                        zone_id[p]    = zid
                        zone_pivot[p] = p

            # start new segment
            if s in ("Accumulation", "Distribution"):
                cur, seg_start = s, i
            else:
                cur, seg_start = None, None

    out["ZoneLow"]      = zone_low
    out["ZoneHigh"]     = zone_high
    out["ZoneType"]     = zone_type
    out["ZoneId"]       = zone_id
    out["ZonePivotIdx"] = zone_pivot
    return out






# ------------------------------------------------
# 5. Track Zone Lifecycle (fixed)
# ------------------------------------------------
def track_zone_lifecycle(df: pd.DataFrame,
                         open_col="Open", close_col="Close",
                         high_col="High", low_col="Low",
                         zone_low_col="ZoneLow", 
                         zone_high_col="ZoneHigh",
                         zone_type_col="ZoneType", 
                         zone_id_col="ZoneId",
                         pivot_idx_col="ZonePivotIdx") -> pd.DataFrame:
    """
    Lifecycle tracker with extended rules:
    - Pending: new zones created at pivot
    - Active: confirmed zones
    - Invalidated: 
        * Rule 1: hard edge break (always ends zone)
        * Rule 2: wrong-colored candle touches zone, later opposite close invalidates
    - Lifecycle column tracks full history (pivot_idx, confirmed_at, end_idx, status, end_reason).
    """

    out = df.copy()
    n = len(out)

    # lifecycle outputs
    zone_pending     = [[] for _ in range(n)]
    zone_active      = [[] for _ in range(n)]
    zone_invalidated = [[] for _ in range(n)]
    zone_all         = [[] for _ in range(n)]

    # trackers
    active_pending   = []
    active_confirmed = []
    active_invalid   = []
    lifecycle        = {}

    for i in range(n):
        o, c, h, l = out[open_col].iloc[i], out[close_col].iloc[i], out[high_col].iloc[i], out[low_col].iloc[i]

        # --- new zone appears
        zid, ztype = out[zone_id_col].iloc[i], out[zone_type_col].iloc[i]
        if pd.notna(zid) and ztype in ("Support", "Resistance"):
            zlow, zhigh = out[zone_low_col].iloc[i], out[zone_high_col].iloc[i]
            z = {"id": zid, "type": ztype, "low": zlow, "high": zhigh,
                 "pivot_idx": i, "confirmed_at": None, "end_idx": None,
                 "status": "Pending", "end_reason": None, "touched": False}
            active_pending.append(z)
            lifecycle[zid] = z.copy()

        # --- check pending zones for invalidation OR confirmation
        still_pending = []
        for z in active_pending:
            zid, ztype, zlow, zhigh = z["id"], z["type"], z["low"], z["high"]

            if ztype == "Support":
                if i > z["pivot_idx"] and c < o and z["confirmed_at"] is None:
                    z.update(status="Invalid", end_idx=i, end_reason="early_fail")
                    active_invalid.append(z.copy()); lifecycle[zid] = z.copy()
                    continue
                if i > z["pivot_idx"] and c > zhigh and c > o and z["confirmed_at"] is None:
                    z["confirmed_at"] = i; lifecycle[zid] = z.copy(); still_pending.append(z); continue

            elif ztype == "Resistance":
                if i > z["pivot_idx"] and c > o and z["confirmed_at"] is None:
                    z.update(status="Invalid", end_idx=i, end_reason="early_fail")
                    active_invalid.append(z.copy()); lifecycle[zid] = z.copy()
                    continue
                if i > z["pivot_idx"] and c < zlow and c < o and z["confirmed_at"] is None:
                    z["confirmed_at"] = i; lifecycle[zid] = z.copy(); still_pending.append(z); continue

            still_pending.append(z)
        active_pending = still_pending

        # --- activate zones when confirmed
        still_pending = []
        for z in active_pending:
            if z["confirmed_at"] is not None and i > z["confirmed_at"]:
                z["status"] = "Active"; lifecycle[z["id"]] = z.copy()
                active_confirmed.append(z.copy()); continue
            still_pending.append(z)
        active_pending = still_pending

        # --- check active zones for hard/soft invalidation
        still_active = []
        for z in active_confirmed:
            zid, ztype, zlow, zhigh = z["id"], z["type"], z["low"], z["high"]

            if ztype == "Resistance":
                # Rule 1: hard break above zone high
                if h >= zhigh:
                    z.update(status="Invalid", end_idx=i, end_reason="break")
                    active_invalid.append(z.copy()); lifecycle[zid] = z.copy(); continue
                # Rule 2: soft rejection
                if not z["touched"] and c > o and h >= zlow:
                    z["touched"] = True; lifecycle[zid] = z.copy()
                if z["touched"] and c < zlow:
                    z.update(status="Invalid", end_idx=i, end_reason="rejection")
                    active_invalid.append(z.copy()); lifecycle[zid] = z.copy(); continue

            elif ztype == "Support":
                # Rule 1: hard break below zone low
                if l <= zlow:
                    z.update(status="Invalid", end_idx=i, end_reason="break")
                    active_invalid.append(z.copy()); lifecycle[zid] = z.copy(); continue
                # Rule 2: soft rejection
                if not z["touched"] and c < o and l <= zhigh:
                    z["touched"] = True; lifecycle[zid] = z.copy()
                if z["touched"] and c > zhigh:
                    z.update(status="Invalid", end_idx=i, end_reason="rejection")
                    active_invalid.append(z.copy()); lifecycle[zid] = z.copy(); continue

            still_active.append(z)
        active_confirmed = still_active

        # --- snapshot
        zone_pending[i]     = [{"id": z["id"], "type": z["type"], "low": z["low"], "high": z["high"]} for z in active_pending]
        zone_active[i]      = list(active_confirmed)
        zone_invalidated[i] = list(active_invalid)
        zone_all[i]         = list(lifecycle.values())

    # attach to df
    out["ZonePending"]     = zone_pending
    out["ZoneActive"]      = zone_active
    out["ZoneInvalidated"] = zone_invalidated
    out["ZoneLifecycle"]   = zone_all
    return out

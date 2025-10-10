# reward_functions/fixed_tp_sl.py

def compute_reward(event):
    """
    Simple event-driven reward:
        +1 → take-profit (price +2%)
        -1 → stop-loss (price -1%)
         0 → hold or waiting

    event: str, one of ["tp", "sl", "hold", "wait"]
    """
    if event == "tp":
        return 1.0
    elif event == "sl":
        return -1.0
    else:
        return 0.0

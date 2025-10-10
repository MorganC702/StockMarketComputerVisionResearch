def compute_reward(
    action, 
    last_action, 
    net_pnl, 
    unrealized_pnl, 
    peak_unrealized_pnl,
    starting_balance,
    *,
    win_reward=10,
    loss_penalty=-1,
    hold_winner_reward=0,
    drawdown_trigger=40.0,
    drawdown_penalty=-5
):
    """
    Simple discrete reward shaping:
      +10 → profitable exit
      -1  → losing exit or holding in loss
       0  → holding in profit
      -5  → if drawdown > $40 from peak
    """
    # --- Exit logic ---
    if action != last_action:
        if net_pnl > 0:
            return win_reward
        else:
            return loss_penalty

    # --- Holding logic ---
    if unrealized_pnl > 0:
        reward = hold_winner_reward
    else:
        reward = loss_penalty

    # --- Drawdown penalty ---
    drawdown = max(0.0, peak_unrealized_pnl - unrealized_pnl)
    if drawdown > drawdown_trigger:
        reward += drawdown_penalty

    return reward

def compute_reward(
    action,
    last_action,
    net_pnl,
    unrealized_pnl,
    peak_unrealized_pnl,
    starting_balance,
    *,
    hold_gain_weight=2000.0,     # reward for positive holds
    underwater_drag=2000.0,      # linear penalty when underwater
    realized_scale=4000.0,       # reward scale on exit
    giveback_strength=1000.0     # quadratic drawdown penalty from peak
):
    # --- Exit logic (position flip) ---
    if action != last_action:
        return realized_scale * (net_pnl / starting_balance)

    # --- Holding logic ---
    peak = max(peak_unrealized_pnl, unrealized_pnl)
    drawdown = max(0.0, peak - unrealized_pnl)

    # 1. Reward for positive unrealized PnL
    hold_reward = hold_gain_weight * max(0.0, unrealized_pnl) / starting_balance

    # 2. Linear penalty if in red
    red_drag = underwater_drag * max(0.0, -unrealized_pnl) / starting_balance

    # 3. Quadratic drawdown penalty from peak
    giveback_penalty = 0.0
    if drawdown > 0:
        # Quadratic curve from peak
        drawdown_ratio = drawdown / (abs(peak) + 1e-9)  # Normalized
        giveback_penalty = giveback_strength * (drawdown_ratio ** 2) * (abs(peak) / starting_balance)

    # Total reward
    reward = hold_reward - red_drag - giveback_penalty
    return reward

import math

def compute_reward_1to1(
    action,
    last_action,
    net_pnl,                 # realized PnL realized this step (incl. fees if you model them here)
    unrealized_pnl,          # current step unrealized PnL
    peak_unrealized_pnl,     # running max since entry *up to this step*
    starting_balance,
    *,
    giveback_threshold=40.0, # start punishing after giving back > $40
    giveback_softness=10.0,  # smoothness ($) around threshold; smaller = sharper
    hold_gain_weight=1.0,    # how much to reward being green
    underwater_drag=0.1,     # bleed when red (per $ of unrealized loss)
    realized_scale=1.0       # scale realized PnL reward on flips
):
    """
    Reward uses only info available at time t. No future leakage.

    - Reward holding winners: +hold_gain_weight * (unrealized_pnl / balance)_+
    - Punish give-back beyond threshold: smooth penalty on drawdown from peak
    - Small drag when underwater to avoid 'freeze-and-pray'
    - On action change (exit/reverse), reward realized PnL proportionally
    """

    # --- Flip/exit: reward realized performance (magnitude matters) ---
    if action != last_action:
        return realized_scale * (net_pnl / starting_balance)

    # --- Holding: compute trailing drawdown from current peak (no peek ahead) ---
    peak = max(peak_unrealized_pnl, unrealized_pnl)  # ensure peak is updated *with* current info only
    drawdown = max(0.0, peak - unrealized_pnl)

    # Reward for being green (encourages riding winners)
    hold_reward = hold_gain_weight * max(0.0, unrealized_pnl) / starting_balance

    # Smooth penalty once give-back exceeds threshold (softplus-like ramp)
    # softplus(x) ~ max(0, x) but smooth; here we center at threshold and scale by softness
    x = (drawdown - giveback_threshold) / max(1e-9, giveback_softness)
    giveback_penalty = math.log1p(math.exp(x)) * giveback_softness  # smooth ramp, ~0 below threshold
    giveback_penalty = giveback_penalty if drawdown > 0 else 0.0

    # Underwater time drag (prevents freezing while red)
    red_drag = underwater_drag * max(0.0, -unrealized_pnl) / starting_balance

    # Combine (note penalty signs)
    reward = hold_reward - (giveback_penalty / starting_balance) - red_drag
    return reward

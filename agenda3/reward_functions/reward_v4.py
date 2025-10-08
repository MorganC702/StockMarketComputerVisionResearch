# reward_function.py

def compute_reward_1to1(
    action, 
    last_action, 
    net_pnl, 
    unrealized_pnl, 
    starting_balance
):
    """
    Compute the reward for a trading step.

    Handles both:
      - trade exit (when action != last_action)
      - holding (when action == last_action)
    """
    if action != last_action:
        pct_return = net_pnl / starting_balance
        return 5.0 if pct_return > 0 else -1.0
    else:
        # Holding → unrealized PnL relative to starting balance
        unrealized_return = unrealized_pnl / starting_balance
        return 1 if unrealized_return >= 0 else -1

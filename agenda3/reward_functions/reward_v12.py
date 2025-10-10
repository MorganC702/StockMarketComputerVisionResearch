def compute_reward(event: str) -> float:
    """
    Event-based reward shaping for TP/SL trading.
    Encourages profit-taking and penalizes losses.
    """
    if event == "tp":
        return 3.0          # strong reward for hitting take profit
    elif event == "sl":
        return -1.0          # penalty for stop loss
    # elif event == "hold":
    #     return -0.1          # mild time decay (discourages holding forever)
    # elif event.startswith("enter"):
    #     return -0.05         # tiny cost to enter a trade (simulated fee/slippage)
    else:
        return 0.0           # no-op for waiting

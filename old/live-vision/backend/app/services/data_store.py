from collections import deque

# Keep last N bars in memory
MAX_BARS = 1000
bar_buffer = deque(maxlen=MAX_BARS)

def add_bar(bar: dict):
    print(f"➕ Added bar at {bar['t']}: o={bar['o']} c={bar['c']}")
    bar_buffer.append(bar)

def get_bars(n: int = 5):
    return list(bar_buffer)[-n:]

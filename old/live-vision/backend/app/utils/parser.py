import pandas as pd

def parse_polygon_payload(data: dict) -> pd.DataFrame:
    
    bars = data.get("bars", [])
    df = pd.DataFrame(bars)
    df["timestamp"] =pd.to_datetime(df["t"], unit='ms')
    return df[["o", "h", "l", "c", "v"]].rename(columns={
        "o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume"
    })
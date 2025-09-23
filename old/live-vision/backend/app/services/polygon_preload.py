from datetime import date, timedelta
from polygon import RESTClient
from app.config import POLYGON_API_KEY
from app.services.data_store import add_bar

import ssl
import certifi
import urllib3

# 🔒 Globally override urllib3 default SSL context
urllib3.util.ssl_.DEFAULT_CIPHERS = "HIGH:!DH:!aNULL"
ssl_context = ssl.create_default_context(cafile=certifi.where())
urllib3.PoolManager(ssl_context=ssl_context)


def preload_bars(ticker="SPY", limit=1000):
    """
    Preload historical bars from Polygon REST API and push into in-memory store.
    Certifi patch ensures SSL verification works properly.
    """
    client = RESTClient(api_key=POLYGON_API_KEY)
    bars = []

    end = date.today()
    start = end - timedelta(days=5)

    for a in client.list_aggs(
        ticker=ticker,
        multiplier=1,
        timespan="minute",
        from_=start.isoformat(),
        to=end.isoformat(),
        limit=limit,  # polygon will page automatically
    ):
        bar = {
            "ticker": ticker,
            "t": a.timestamp,
            "o": a.open,
            "h": a.high,
            "l": a.low,
            "c": a.close,
            "v": a.volume,
        }
        bars.append(bar)

    # Push into your in-memory buffer
    for bar in bars[-limit:]:
        add_bar(bar)

    print(f"🔥 Preloaded {len(bars[-limit:])} {ticker} bars")
    return bars[-limit:]

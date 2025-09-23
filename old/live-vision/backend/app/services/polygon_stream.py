import asyncio
from polygon import WebSocketClient
from polygon.websocket.models import WebSocketMessage, Feed, Market

from typing import List
from app.config import POLYGON_API_KEY, POLYGON_SUBSCRIPTIONS
from app.services.data_store import add_bar

from polygon.websocket.models import EquityAgg

def handle_msg(msgs: list):
    for m in msgs:
        if isinstance(m, EquityAgg):
            bar = {
                "ticker": m.symbol,
                "t": m.end_timestamp,   # epoch ms
                "o": m.open,
                "h": m.high,
                "l": m.low,
                "c": m.close,
                "v": m.volume,
            }
            print(f"📩 New bar: {bar}")  # debug
            add_bar(bar)


async def start_polygon_ws():
    ws = WebSocketClient(
        api_key=POLYGON_API_KEY,
        feed=Feed.Delayed,   # free tier
        market=Market.Stocks
    )

    print("🔌 Connecting to Polygon WebSocket...")

    for sub in POLYGON_SUBSCRIPTIONS:
        ws.subscribe(sub)

    print(f"✅ Subscribed to {POLYGON_SUBSCRIPTIONS}")
 
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, ws.run, handle_msg)
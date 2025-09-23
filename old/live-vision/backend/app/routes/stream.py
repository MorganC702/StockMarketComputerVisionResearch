from fastapi import APIRouter
from fastapi.responses import StreamingResponse
import asyncio
import json
import pandas as pd
import base64

from app.services.data_store import get_bars
from app.services.chart_generator import generate_chart
from app.services.model_infer import run_inference

router = APIRouter()

async def event_stream(window: int = 5):
    while True:
        print("Stream Hit")
        bars = get_bars(window)

        if len(bars) >= window:
            # Convert bars → DataFrame
            df = pd.DataFrame(bars).rename(columns={
                "o": "Open", "h": "High", "l": "Low",
                "c": "Close", "v": "Volume", "t": "Timestamp",
            })
            df.set_index(pd.to_datetime(df["Timestamp"], unit="ms"), inplace=True)

            # 🖼️ Generate chart image
            image_buf = generate_chart(df)

            # 🧠 Run inference
            preds = run_inference(image_buf)

            # Encode chart to base64
            img_b64 = base64.b64encode(image_buf.getvalue()).decode("utf-8")
            img_uri = f"data:image/png;base64,{img_b64}"

            # Final payload
            payload = {
                "image": img_uri,
                "candles": bars,
                "predictions": preds,
            }

            # Debug
            print("📊 Sending window:", bars)
            print("🤖 Predictions:", preds)

            yield f"data: {json.dumps(payload)}\n\n"

        await asyncio.sleep(10)

@router.get("/sse")
async def stream(window: int = 5):
    return StreamingResponse(event_stream(window), media_type="text/event-stream")

from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.routes import stream
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
import asyncio
from app.routes import stream
from app.services.polygon_stream import start_polygon_ws
from app.services.polygon_preload import preload_bars


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ✅ Startup
    preload_bars("SPY", limit=1000)   # preload history
    asyncio.create_task(start_polygon_ws())  # start websocket in background
    yield
    # ✅ Shutdown (optional clean-up)
    print("👋 Shutting down...")

app = FastAPI(
    title="Stock Vision API",
    description="Processes Polygon webhook -> renders chart -> YOLO inference",
    version="0.1.0",
    lifespan=lifespan, 
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],   # important so OPTIONS requests don’t 405
    allow_headers=["*"],
)

app.include_router(stream.router, prefix="/stream")


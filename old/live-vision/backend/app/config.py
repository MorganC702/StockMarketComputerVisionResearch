import os
from dotenv import load_dotenv

# Load variables from .env
load_dotenv()

POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")
POLYGON_SUBSCRIPTIONS = os.getenv("POLYGON_SUBSCRIPTIONS", "AM.SPY").split(",")
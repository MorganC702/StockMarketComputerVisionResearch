from io import BytesIO
from ultralytics import YOLO
from PIL import Image
import numpy as np
import os

# ✅ Path relative to model_infer.py
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best.pt")

# Load YOLO model
model = YOLO(MODEL_PATH)

# Optional: label mapping
LABEL_MAP = {
    0: "High Bearish Sentiment",
    1: "Medium-High Bearish Sentiment",
    2: "Neutral Bearish Sentiment",
    3: "Medium-Low Bearish Sentiment",
    4: "Low Bearish Sentiment",
    5: "High Bullish Sentiment",
    6: "Medium-High Bullish Sentiment",
    7: "Neutral Bullish Sentiment",
    8: "Medium-Low Bullish Sentiment",
    9: "Low Bullish Sentiment",
    10: "Doji",
}

def run_inference(image_buf: BytesIO):
    image = Image.open(image_buf).convert("RGB")
    np_img = np.array(image)

    # returns a list of Results (usually length 1 if you pass a single image)
    results_list = model.predict(np_img, verbose=False)

    predictions = []
    for r in results_list:         # loop over each Results object
        for box in r.boxes:        # loop over each detection
            cls_id = int(box.cls.item())
            conf = float(box.conf.item())
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            predictions.append({
                "class_id": cls_id,
                "label": LABEL_MAP.get(cls_id, f"class_{cls_id}"),
                "confidence": round(conf, 4),
                "bbox": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)]
            })

    return predictions

import { useEffect, useState, useRef } from "react";

type Prediction = {
  class_id: number;
  label: string;
  confidence: number;
  bbox: [number, number, number, number]; // [x1, y1, x2, y2]
};

export default function CandlestickChartView() {
  const [imgSrc, setImgSrc] = useState<string | null>(null);
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const eventSource = new EventSource("http://127.0.0.1:8000/stream/sse");

    eventSource.onmessage = (event) => {
      const data = JSON.parse(event.data);

      if (data.image) {
        setImgSrc(data.image);
      }

      if (Array.isArray(data.predictions) && data.predictions.length > 0) {
        setPredictions(data.predictions);
      } else {
        setPredictions([]);
      }
    };

    eventSource.onerror = (err) => {
      console.error("SSE error:", err);
      eventSource.close();
    };

    return () => eventSource.close();
  }, []);

  useEffect(() => {
    if (!canvasRef.current || !imgSrc) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const image = new Image();
    image.src = imgSrc;
    image.onload = () => {
      const W = canvas.width;
      const H = canvas.height;
      const PADDING = 60; // extra margin for labels

      const drawW = W - 2 * PADDING;
      const drawH = H - 2 * PADDING;

      ctx.clearRect(0, 0, W, H);

      // --- Draw image inside padded area ---
      ctx.drawImage(image, PADDING, PADDING, drawW, drawH);

      // --- Compute scaling factors ---
      const scaleX = drawW / image.width;
      const scaleY = drawH / image.height;

      predictions.forEach((p) => {
        const [x1, y1, x2, y2] = p.bbox;

        // Scale + shift into padded canvas space
        const x1s = PADDING + x1 * scaleX;
        const y1s = PADDING + y1 * scaleY;
        const x2s = PADDING + x2 * scaleX;
        const y2s = PADDING + y2 * scaleY;

        const w = x2s - x1s;
        const h = y2s - y1s;

        // Color based on label
        let stroke = "blue";
        if (p.label.toLowerCase().includes("bull")) stroke = "green";
        if (p.label.toLowerCase().includes("bear")) stroke = "red";

        ctx.strokeStyle = stroke;
        ctx.lineWidth = 2;
        ctx.strokeRect(x1s, y1s, w, h);

        // --- Label ---
        const text = `${p.label} ${(p.confidence * 100).toFixed(1)}%`;
        ctx.font = "12px sans-serif";
        const textW = ctx.measureText(text).width;

        // Clamp so text stays inside canvas
        const labelX = Math.max(PADDING, Math.min(W - textW - PADDING, x1s));
        const labelY = Math.max(PADDING + 16, y1s);

        // background
        ctx.fillStyle = "rgba(255,255,255,0.7)";
        ctx.fillRect(labelX, labelY - 16, textW + 8, 16);

        // text
        ctx.fillStyle = "black";
        ctx.fillText(text, labelX + 4, labelY - 4);
      });
    };
  }, [imgSrc, predictions]);

  return (
    <div className="min-h-screen bg-blue-900 flex items-center justify-center p-6">
      {imgSrc ? (
        <canvas
          ref={canvasRef}
          width={640}
          height={640}
          className="border border-gray-200 rounded-xl"
        />
      ) : (
        <p className="text-gray-200 text-lg">Waiting for chart...</p>
      )}
    </div>
  );
}

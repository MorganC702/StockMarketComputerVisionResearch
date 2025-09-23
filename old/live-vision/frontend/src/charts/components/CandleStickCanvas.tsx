import { useEffect, useRef } from "react";

type Candle = {
  t: number; // timestamp
  o: number; // open
  h: number; // high
  l: number; // low
  c: number; // close
  v: number; // volume
};

type Prediction = {
  class_id: number;
  label: string;
  confidence: number;
  bbox: [number, number, number, number]; // [x1, y1, x2, y2]
};

type Props = {
  candles: Candle[];
  predictions?: Prediction[];
};

export default function CandlestickCanvas({ candles, predictions = [] }: Props) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    if (!canvasRef.current || candles.length === 0) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const W = 640;
    const H = 400;
    const PADDING = 40;

    ctx.clearRect(0, 0, W, H);

    // --- Candle scaling ---
    const minPrice = Math.min(...candles.map((c) => c.l));
    const maxPrice = Math.max(...candles.map((c) => c.h));
    const ySpan = maxPrice - minPrice;

    const xStep = (W - 2 * PADDING) / candles.length;
    const bodyWidth = Math.max(4, xStep * 0.6);
    const wickWidth = 2;

    const yMap = (val: number) =>
      H - PADDING - ((val - minPrice) / ySpan) * (H - 2 * PADDING);

    // --- Draw candles ---
    candles.forEach((c, i) => {
      const xCenter = PADDING + i * xStep + xStep / 2;
      const x0 = xCenter - bodyWidth / 2;
      // const x1 = xCenter + bodyWidth / 2;

      const yHigh = yMap(c.h);
      const yLow = yMap(c.l);
      const yOpen = yMap(c.o);
      const yClose = yMap(c.c);

      const isBullish = c.c > c.o;
      ctx.fillStyle = isBullish ? "green" : "red";
      ctx.strokeStyle = ctx.fillStyle;

      // Wick
      ctx.beginPath();
      ctx.moveTo(xCenter, yHigh);
      ctx.lineTo(xCenter, yLow);
      ctx.lineWidth = wickWidth;
      ctx.stroke();

      // Body
      ctx.fillRect(
        x0,
        Math.min(yOpen, yClose),
        bodyWidth,
        Math.abs(yClose - yOpen)
      );
    });

    // --- Draw predictions (if they’re in chart coordinates) ---
    predictions.forEach((p) => {
      const [x1Raw, y1Raw, x2Raw, y2Raw] = p.bbox;

      // Scale/shift into padded chart space
      const x1 = PADDING + (x1Raw / W) * (W - 2 * PADDING);
      const x2 = PADDING + (x2Raw / W) * (W - 2 * PADDING);
      const y1 = PADDING + (y1Raw / H) * (H - 2 * PADDING);
      const y2 = PADDING + (y2Raw / H) * (H - 2 * PADDING);

      let stroke = "blue";
      if (p.label.toLowerCase().includes("bull")) stroke = "green";
      if (p.label.toLowerCase().includes("bear")) stroke = "red";

      ctx.strokeStyle = stroke;
      ctx.lineWidth = 2;
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

      // Label with confidence
      const text = `${p.label} ${(p.confidence * 100).toFixed(1)}%`;
      ctx.font = "12px sans-serif";
      const textW = ctx.measureText(text).width;
      ctx.fillStyle = "rgba(255,255,255,0.7)";
      ctx.fillRect(x1, y1 - 16, textW + 8, 16);

      ctx.fillStyle = "black";
      ctx.fillText(text, x1 + 4, y1 - 4);
    });
  }, [candles, predictions]);

  return (
    <canvas
      ref={canvasRef}
      width={640}
      height={400}
      className="border border-gray-300 rounded"
    />
  );
}

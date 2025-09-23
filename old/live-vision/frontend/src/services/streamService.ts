export type Candle = {
  t: number;
  o: number;
  h: number;
  l: number;
  c: number;
};

export type Prediction = {
  label: string;
  x: number;
  y: number;
  w: number;
  h: number;
};

export function subscribeToCandles(
  url: string,
  onData: (candles: Candle[], preds: Prediction[]) => void
) {
  const eventSource = new EventSource(url);

  eventSource.onmessage = (event) => {
    const data = JSON.parse(event.data);
    if (data.candles) {
      onData(data.candles, data.predictions || []);
    }
  };

  eventSource.onerror = (err) => {
    console.error("SSE error:", err);
    eventSource.close();
  };

  return () => eventSource.close();
}

import os
from PIL import Image, ImageDraw
import pandas as pd

def generate_images(
    candle_data: dict,
    break_levels: dict,
    break_events: dict,
    output_root: str = "dataset",
    image_size: tuple = (640, 640),
    dot_radius: int = 3,
    max_trail_length: int = 100,
    price_scale: float = 16.0  # pixels per price unit
):
    os.makedirs(output_root, exist_ok=True)

    base_tf = sorted(candle_data.keys())[0]
    base_df = candle_data[base_tf]
    timestamps = list(base_df.index)

    print(f"🧠 Using base TF: {base_tf}, Rendering {len(timestamps)} frames")

    for i, ts in enumerate(timestamps):
        price_now = base_df.loc[ts]["Close"]
        trail_df = base_df.iloc[max(0, i - max_trail_length):i]

        for tf in candle_data.keys():
            tf_dir = os.path.join(output_root, tf)
            img_dir = os.path.join(tf_dir, "images")
            os.makedirs(img_dir, exist_ok=True)

            out_file = os.path.join(img_dir, f"img_{ts.strftime('%Y-%m-%d_%H-%M')}.png")

            # Create blank white image
            img = Image.new("RGB", image_size, (255, 255, 255))
            draw = ImageDraw.Draw(img)

            x_center = image_size[0] // 2
            y_center = image_size[1] // 2

            # Determine visible price range based on fixed scale
            half_viewport = image_size[1] / (2 * price_scale)
            price_min = price_now - half_viewport
            price_max = price_now + half_viewport

            # --------- 🟩🟥 Draw Active Break Levels ----------
            levels_df = break_levels.get(tf)
            events_df = break_events.get(tf)

            if levels_df is not None and not levels_df.empty:
                formed_levels = levels_df[levels_df["level_time"] <= ts]

                for _, row in formed_levels.iterrows():
                    level_price = row["level_price"]
                    level_type = row["type"]
                    level_time = row["level_time"]

                    # Skip broken levels
                    broken = False
                    if events_df is not None and not events_df.empty:
                        match = events_df[
                            (events_df["level_price"] == level_price) &
                            (events_df["break_type"] == level_type) &
                            (events_df["break_time"] <= ts)
                        ]
                        if not match.empty:
                            broken = True
                    if broken:
                        continue

                    # Skip levels outside of vertical viewport
                    if not (price_min <= level_price <= price_max):
                        continue

                    try:
                        level_time_idx = base_df.index.get_loc(level_time)
                    except KeyError:
                        continue

                    frames_since_level = i - level_time_idx
                    if frames_since_level < 0:
                        continue

                    # ✅ Sync with dot trail movement
                    x_start = x_center - frames_since_level * 2 * dot_radius
                    x_start = max(x_start, 0)

                    y = y_center - (level_price - price_now) * price_scale
                    color = (255, 0, 0) if level_type == "high_break" else (0, 200, 0)

                    draw.line([(x_start, y), (image_size[0], y)], fill=color, width=2)

            # --------- ⚫ Fading Trail Dots ----------
            trail_len = len(trail_df)
            for j, (_, row) in enumerate(trail_df.iterrows()):
                trail_price = row["Close"]
                dy = (trail_price - price_now) * price_scale
                cx = x_center - (trail_len - j) * 2 * dot_radius
                cy = y_center - dy

                gray_val = int(255 * (1 - (j + 1) / trail_len))
                gray_val = max(0, min(255, gray_val))
                color = (gray_val, gray_val, gray_val)

                draw.ellipse(
                    [cx - dot_radius, cy - dot_radius, cx + dot_radius, cy + dot_radius],
                    fill=color
                )

            # --------- ⚫ Center Dot ----------
            draw.ellipse(
                [
                    x_center - dot_radius,
                    y_center - dot_radius,
                    x_center + dot_radius,
                    y_center + dot_radius,
                ],
                fill=(0, 0, 0)
            )

            img.save(out_file)

        if i % 100 == 0:
            print(f"🖼️ Rendered {i}/{len(timestamps)} frames")

    print("✅ Done rendering images.")

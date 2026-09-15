import argparse
import csv
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from test_manual_tiles import DEFAULT_DATA_ROOT, collect_valid_sample_ids, discover_regions


STAGE_LABELS = [
    ("w1", "播种/出苗期"),
    ("w2", "苗期/拔节期"),
    ("w3", "快速生长期"),
    ("w4", "抽雄吐丝期"),
    ("w5", "灌浆期"),
    ("w6", "成熟/收获期"),
]


def read_period_rgb(region_dir, sample_name, period):
    with rasterio.open(region_dir / f"w{period}" / sample_name) as src:
        data = src.read().astype(np.float32)
    # Per-period bands: B2, B3, B4, B8, B11, B12.
    return np.stack([data[2], data[1], data[0]], axis=-1)


def joint_stretch(period_rgbs, low=2, high=98):
    stacked = np.stack(period_rgbs, axis=0)
    output = []
    limits = []
    for channel in range(3):
        values = stacked[..., channel]
        valid = values[np.isfinite(values) & (values > 0)]
        if valid.size:
            lo, hi = np.percentile(valid, [low, high])
        else:
            lo, hi = 0.0, 1.0
        if hi <= lo:
            hi = lo + 1e-6
        limits.append((lo, hi))

    for rgb in period_rgbs:
        stretched = np.zeros_like(rgb, dtype=np.uint8)
        for channel, (lo, hi) in enumerate(limits):
            values = np.clip((rgb[..., channel] - lo) / (hi - lo), 0, 1)
            stretched[..., channel] = (values * 255).astype(np.uint8)
        output.append(stretched)
    return output


def publication_stretch(rgb, low=2, high=98):
    stretched = np.zeros_like(rgb, dtype=np.float32)
    for channel in range(3):
        values = rgb[..., channel]
        valid = values[np.isfinite(values) & (values > 0)]
        if valid.size:
            lo, hi = np.percentile(valid, [low, high])
        else:
            lo, hi = 0.0, 1.0
        if hi <= lo:
            hi = lo + 1e-6
        stretched[..., channel] = np.clip((values - lo) / (hi - lo), 0, 1)

    valid_pixels = np.any(rgb > 0, axis=-1)
    if np.any(valid_pixels):
        means = np.asarray(
            [stretched[..., c][valid_pixels].mean() for c in range(3)],
            dtype=np.float32,
        )
        target = float(np.mean(means))
        gains = np.clip(target / np.maximum(means, 1e-6), 0.82, 1.18)
        # Thin cloud and atmospheric haze often over-amplify the blue channel.
        gains[2] = min(gains[2], 1.02)
        stretched *= gains.reshape(1, 1, 3)

    # Mild contrast and saturation tuning for clean publication visualization.
    stretched = np.clip((stretched - 0.5) * 1.05 + 0.5, 0, 1)
    gray = stretched.mean(axis=2, keepdims=True)
    stretched = np.clip(gray + (stretched - gray) * 0.90, 0, 1)
    stretched[~valid_pixels] = 0
    return (stretched * 255).astype(np.uint8)


def blue_artifact_score(rgb):
    values = rgb.astype(np.float32)
    red, green, blue = values[..., 0], values[..., 1], values[..., 2]
    valid = np.any(values > 0, axis=-1)
    blue_excess = (blue > red * 1.22) & (blue > green * 1.12) & (blue > 70)
    if not np.any(valid):
        return 1.0
    return float(np.mean(blue_excess[valid]))


def load_font(size, bold=False):
    candidates = [
        Path("C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def make_figure(period_rgbs, panel_size, show_labels):
    panels = [
        Image.fromarray(rgb).resize((panel_size, panel_size), Image.Resampling.LANCZOS)
        for rgb in period_rgbs
    ]
    gap = max(14, panel_size // 48)
    outer = max(16, panel_size // 40)
    label_h = max(78, panel_size // 8) if show_labels else 0
    width = outer * 2 + panel_size * 6 + gap * 5
    height = outer * 2 + label_h + panel_size
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    period_font = load_font(max(24, panel_size // 28), bold=True)
    stage_font = load_font(max(22, panel_size // 32))

    for index, (panel, (period, stage)) in enumerate(zip(panels, STAGE_LABELS)):
        x = outer + index * (panel_size + gap)
        if show_labels:
            period_box = draw.textbbox((0, 0), period, font=period_font)
            stage_box = draw.textbbox((0, 0), stage, font=stage_font)
            period_w = period_box[2] - period_box[0]
            stage_w = stage_box[2] - stage_box[0]
            draw.text(
                (x + (panel_size - period_w) // 2, outer),
                period,
                fill=(15, 15, 15),
                font=period_font,
            )
            draw.text(
                (x + (panel_size - stage_w) // 2, outer + label_h // 2),
                stage,
                fill=(35, 35, 35),
                font=stage_font,
            )
        canvas.paste(panel, (x, outer + label_h))
    return canvas


def main():
    parser = argparse.ArgumentParser(
        description="Generate publication-ready six-period Sentinel-2 comparison figures."
    )
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--regions", nargs="*", default=None)
    parser.add_argument("--output-dir", default="manual_tile_six_period_sentinel_all")
    parser.add_argument("--panel-size", type=int, default=768)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--visual-mode",
        choices=["publication", "joint"],
        default="publication",
        help="publication independently balances each period; joint preserves reflectance comparability.",
    )
    parser.add_argument(
        "--show-labels",
        action="store_true",
        help="Show w1-w6 and phenology labels above panels.",
    )
    args = parser.parse_args()

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = []

    for region_dir in discover_regions(args.data_root, args.regions):
        region_out = output_root / region_dir.name
        region_out.mkdir(parents=True, exist_ok=True)
        for sample_name in tqdm(
            collect_valid_sample_ids(region_dir), desc=region_dir.name
        ):
            period_rgbs = [
                read_period_rgb(region_dir, sample_name, period)
                for period in range(1, 7)
            ]
            if args.visual_mode == "joint":
                period_rgbs = joint_stretch(period_rgbs)
            else:
                period_rgbs = [publication_stretch(rgb) for rgb in period_rgbs]
            blue_scores = [blue_artifact_score(rgb) for rgb in period_rgbs]
            figure = make_figure(period_rgbs, args.panel_size, args.show_labels)
            stem = Path(sample_name).stem
            output_path = region_out / f"{stem}_six_period_sentinel.png"
            figure.save(output_path, dpi=(args.dpi, args.dpi), compress_level=4)
            manifest.append(
                {
                    "region": region_dir.name,
                    "sample": sample_name,
                    "figure": str(output_path.resolve()),
                    "max_blue_artifact_ratio": max(blue_scores),
                    "mean_blue_artifact_ratio": float(np.mean(blue_scores)),
                }
            )

    manifest_path = output_root / "six_period_sentinel_manifest.csv"
    with manifest_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=manifest[0].keys())
        writer.writeheader()
        writer.writerows(
            sorted(manifest, key=lambda row: row["max_blue_artifact_ratio"])
        )
    print(f"[INFO] Created {len(manifest)} figures: {output_root.resolve()}")


if __name__ == "__main__":
    main()

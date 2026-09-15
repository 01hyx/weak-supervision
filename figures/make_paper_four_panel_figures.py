import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import rasterio
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from test_manual_tiles import (
    DEFAULT_DATA_ROOT,
    IGNORE_LABEL,
    collect_valid_sample_ids,
    confusion_matrix,
    discover_regions,
    load_sample,
    metrics_from_cm,
)


def percentile_stretch(channel, low=2, high=98):
    valid = channel[np.isfinite(channel) & (channel > 0)]
    if valid.size == 0:
        return np.zeros_like(channel, dtype=np.uint8)
    lo, hi = np.percentile(valid, [low, high])
    if hi <= lo:
        hi = lo + 1e-6
    stretched = np.clip((channel - lo) / (hi - lo), 0, 1)
    return (stretched * 255).astype(np.uint8)


def make_rgb(image_chw, time_index):
    start = time_index * 6
    # Bands in each period are [B2, B3, B4, B8, B11, B12].
    return np.stack(
        [
            percentile_stretch(image_chw[start + 2]),
            percentile_stretch(image_chw[start + 1]),
            percentile_stretch(image_chw[start + 0]),
        ],
        axis=-1,
    )


def read_probability(path):
    with rasterio.open(path) as src:
        prob = src.read(1).astype(np.float32)
    if prob.max(initial=0) > 1.0:
        prob /= 255.0
    return prob


def binary_mask_image(mask, valid):
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    out[mask & valid] = (255, 255, 255)
    out[~valid] = (90, 90, 90)
    return out


def make_mask_overlay(rgb, mask, valid):
    out = rgb.astype(np.float32)
    target = mask & valid
    # Color-blind-safe blue contrasts clearly with vegetation-heavy green RGB imagery.
    fill_color = np.asarray([0, 114, 178], dtype=np.float32)
    out[target] = 0.50 * out[target] + 0.50 * fill_color

    kernel = np.ones((3, 3), dtype=np.uint8)
    mask_u8 = target.astype(np.uint8)
    boundary = cv2.dilate(mask_u8, kernel) != cv2.erode(mask_u8, kernel)
    out[boundary & valid] = (255, 255, 255)
    out[~valid] = (70, 70, 70)
    return np.clip(out, 0, 255).astype(np.uint8)


def load_font(size, bold=False):
    candidates = [
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/calibrib.ttf" if bold else "C:/Windows/Fonts/calibri.ttf"),
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def resize_panel(array, panel_size, is_mask=False):
    resampling = Image.Resampling.NEAREST if is_mask else Image.Resampling.LANCZOS
    return Image.fromarray(array).resize((panel_size, panel_size), resampling)


def make_figure(rgb, gt, pred, valid, panel_size):
    gt_img = binary_mask_image(gt, valid)
    pred_img = binary_mask_image(pred, valid)
    gt_overlay_img = make_mask_overlay(rgb, gt, valid)
    pred_overlay_img = make_mask_overlay(rgb, pred, valid)

    panels = [
        resize_panel(rgb, panel_size),
        resize_panel(gt_img, panel_size, is_mask=True),
        resize_panel(pred_img, panel_size, is_mask=True),
        resize_panel(gt_overlay_img, panel_size),
        resize_panel(pred_overlay_img, panel_size),
    ]

    gap = max(18, panel_size // 42)
    outer = max(12, panel_size // 64)
    width = outer * 2 + panel_size * 5 + gap * 4
    height = outer * 2 + panel_size
    canvas = Image.new("RGB", (width, height), (255, 255, 255))

    for index, panel in enumerate(panels):
        x = outer + index * (panel_size + gap)
        canvas.paste(panel, (x, outer))
    return canvas


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create publication-ready horizontal five-panel figures for all test samples."
    )
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--prob-root", default="manual_tile_prediction_tifs_finetune_head")
    parser.add_argument("--output-dir", default="manual_tile_paper_five_panel_all")
    parser.add_argument("--regions", nargs="*", default=None)
    parser.add_argument(
        "--sample-csv",
        default=None,
        help="Optional CSV containing region and sample columns; only listed samples are generated.",
    )
    parser.add_argument("--threshold", type=float, default=0.65)
    parser.add_argument("--rgb-time", type=int, default=4, help="0-based period; default is w5.")
    parser.add_argument("--panel-size", type=int, default=1024)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    missing = []
    selected_samples = None
    if args.sample_csv:
        selected_samples = {}
        with Path(args.sample_csv).open("r", encoding="utf-8-sig", newline="") as file:
            for row in csv.DictReader(file):
                selected_samples.setdefault(row["region"], set()).add(row["sample"])

    regions = discover_regions(args.data_root, args.regions)
    for region_dir in regions:
        region = region_dir.name
        if selected_samples is not None and region not in selected_samples:
            continue
        region_out = output_dir / region
        region_out.mkdir(parents=True, exist_ok=True)
        sample_names = collect_valid_sample_ids(region_dir)
        if selected_samples is not None:
            sample_names = [
                sample_name
                for sample_name in sample_names
                if sample_name in selected_samples[region]
            ]

        for sample_name in tqdm(sample_names, desc=region):
            stem = Path(sample_name).stem
            prob_path = Path(args.prob_root) / region / "prob" / f"{stem}_prob.tif"
            if not prob_path.exists():
                missing.append(str(prob_path))
                continue

            image, gt_with_ignore = load_sample(region_dir, sample_name, ignore_nodata=True)
            valid = gt_with_ignore != IGNORE_LABEL
            gt = gt_with_ignore == 1
            prob = read_probability(prob_path)
            pred = (prob >= args.threshold) & valid
            rgb = make_rgb(image, args.rgb_time)

            figure = make_figure(rgb, gt, pred, valid, args.panel_size)
            output_path = region_out / f"{stem}_paper_five_panel.png"
            figure.save(output_path, dpi=(args.dpi, args.dpi), compress_level=4)

            metrics = metrics_from_cm(
                confusion_matrix(pred.astype(np.uint8), gt_with_ignore)
            )
            rows.append(
                {
                    "region": region,
                    "sample": sample_name,
                    "figure": str(output_path.resolve()),
                    "threshold": args.threshold,
                    "oa": metrics["oa"],
                    "miou": metrics["miou"],
                    "target_iou": metrics["iou"][1],
                    "target_f1": metrics["f1"][1],
                }
            )

    manifest_path = output_dir / "paper_five_panel_manifest.csv"
    with manifest_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"[INFO] Created {len(rows)} figures in: {output_dir.resolve()}")
    print(f"[INFO] Manifest: {manifest_path.resolve()}")
    if missing:
        missing_path = output_dir / "missing_probability_tifs.txt"
        missing_path.write_text("\n".join(missing), encoding="utf-8")
        print(f"[WARN] Missing {len(missing)} probability TIFs: {missing_path.resolve()}")


if __name__ == "__main__":
    main()

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import rasterio
from PIL import Image
from tqdm import tqdm

from test_manual_tiles import (
    DEFAULT_DATA_ROOT,
    IGNORE_LABEL,
    confusion_matrix,
    load_sample,
    metrics_from_cm,
)


def percentile_stretch(channel, low=2, high=98):
    values = channel[np.isfinite(channel) & (channel > 0)]
    if values.size == 0:
        return np.zeros_like(channel, dtype=np.uint8)
    lo, hi = np.percentile(values, [low, high])
    if hi <= lo:
        hi = lo + 1e-6
    return (np.clip((channel - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)


def make_rgb(image, time_index=4):
    start = time_index * 6
    return np.stack(
        [
            percentile_stretch(image[start + 2]),
            percentile_stretch(image[start + 1]),
            percentile_stretch(image[start]),
        ],
        axis=-1,
    )


def read_prob(path):
    with rasterio.open(path) as src:
        prob = src.read(1).astype(np.float32)
    if prob.max(initial=0) > 1:
        prob /= 255.0
    return prob


def mask_image(mask, valid):
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    out[mask & valid] = 255
    out[~valid] = 90
    return out


def overlay(rgb, mask, valid):
    out = rgb.astype(np.float32)
    target = mask & valid
    blue = np.asarray([0, 114, 178], dtype=np.float32)
    out[target] = 0.5 * out[target] + 0.5 * blue
    kernel = np.ones((3, 3), dtype=np.uint8)
    edge = cv2.dilate(target.astype(np.uint8), kernel) != cv2.erode(
        target.astype(np.uint8), kernel
    )
    out[edge & valid] = 255
    out[~valid] = 70
    return np.clip(out, 0, 255).astype(np.uint8)


def resize(array, size, nearest=False):
    method = Image.Resampling.NEAREST if nearest else Image.Resampling.LANCZOS
    return Image.fromarray(array).resize((size, size), method)


def make_figure(rgb, gt, before, after, valid, panel_size):
    panels = [
        resize(rgb, panel_size),
        resize(mask_image(gt, valid), panel_size, nearest=True),
        resize(mask_image(before, valid), panel_size, nearest=True),
        resize(mask_image(after, valid), panel_size, nearest=True),
        resize(overlay(rgb, before, valid), panel_size),
        resize(overlay(rgb, after, valid), panel_size),
    ]
    gap = max(18, panel_size // 42)
    outer = max(12, panel_size // 64)
    width = outer * 2 + panel_size * len(panels) + gap * (len(panels) - 1)
    height = outer * 2 + panel_size
    canvas = Image.new("RGB", (width, height), "white")
    for i, panel in enumerate(panels):
        canvas.paste(panel, (outer + i * (panel_size + gap), outer))
    return canvas


def add_cm(total, cm):
    total += cm
    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split", default="manual_tiles_filtered_problem_removed/filtered_trainval_split.json"
    )
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--before-prob-root", default="manual_tile_prediction_tifs_all")
    parser.add_argument(
        "--after-prob-root", default="manual_tile_prediction_tifs_finetune_head"
    )
    parser.add_argument(
        "--output-dir", default="manual_tile_finetune_before_after_filtered_val"
    )
    parser.add_argument("--before-threshold", type=float, default=0.12)
    parser.add_argument("--after-threshold", type=float, default=0.65)
    parser.add_argument("--panel-size", type=int, default=1024)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    split = json.loads(Path(args.split).read_text(encoding="utf-8"))
    samples = split["val"]
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    rows = []
    before_total = np.zeros((2, 2), dtype=np.int64)
    after_total = np.zeros((2, 2), dtype=np.int64)

    for item in tqdm(samples, desc="filtered validation"):
        region = item["region"]
        sample = item["sample"]
        stem = Path(sample).stem
        region_dir = Path(args.data_root) / region
        image, gt_label = load_sample(region_dir, sample, ignore_nodata=True)
        valid = gt_label != IGNORE_LABEL
        gt = gt_label == 1

        before_prob = read_prob(
            Path(args.before_prob_root) / region / "prob" / f"{stem}_prob.tif"
        )
        after_prob = read_prob(
            Path(args.after_prob_root) / region / "prob" / f"{stem}_prob.tif"
        )
        before = (before_prob >= args.before_threshold) & valid
        after = (after_prob >= args.after_threshold) & valid

        before_cm = confusion_matrix(before.astype(np.uint8), gt_label)
        after_cm = confusion_matrix(after.astype(np.uint8), gt_label)
        before_total = add_cm(before_total, before_cm)
        after_total = add_cm(after_total, after_cm)
        before_metrics = metrics_from_cm(before_cm)
        after_metrics = metrics_from_cm(after_cm)

        figure = make_figure(make_rgb(image), gt, before, after, valid, args.panel_size)
        out_dir = out_root / region
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{stem}_finetune_before_after.png"
        figure.save(out_path, dpi=(args.dpi, args.dpi), compress_level=4)

        rows.append(
            {
                "region": region,
                "sample": sample,
                "figure": str(out_path.resolve()),
                "before_threshold": args.before_threshold,
                "after_threshold": args.after_threshold,
                "before_oa": before_metrics["oa"],
                "after_oa": after_metrics["oa"],
                "before_miou": before_metrics["miou"],
                "after_miou": after_metrics["miou"],
                "before_target_iou": before_metrics["iou"][1],
                "after_target_iou": after_metrics["iou"][1],
                "before_target_f1": before_metrics["f1"][1],
                "after_target_f1": after_metrics["f1"][1],
            }
        )

    manifest = out_root / "finetune_before_after_manifest.csv"
    with manifest.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "validation_samples": len(samples),
        "before_threshold": args.before_threshold,
        "after_threshold": args.after_threshold,
        "before_confusion_matrix": before_total.tolist(),
        "after_confusion_matrix": after_total.tolist(),
        "before_metrics": metrics_from_cm(before_total),
        "after_metrics": metrics_from_cm(after_total),
        "panel_order": [
            "Sentinel-2 RGB",
            "ground-truth mask",
            "before-finetuning prediction mask",
            "after-finetuning prediction mask",
            "before-finetuning prediction overlay",
            "after-finetuning prediction overlay",
        ],
    }
    summary_path = out_root / "finetune_before_after_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[INFO] Created {len(rows)} figures: {out_root.resolve()}")
    print(f"[INFO] Summary: {summary_path.resolve()}")


if __name__ == "__main__":
    main()

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image, ImageDraw, ImageFont
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
    out = np.clip((channel - lo) / (hi - lo), 0, 1)
    return (out * 255).astype(np.uint8)


def make_rgb(image_chw, time_index):
    start = time_index * 6
    # 每期 6 个波段顺序为 B2, B3, B4, B8, B11, B12，这里使用真彩色 B4/B3/B2。
    return np.stack(
        [
            percentile_stretch(image_chw[start + 2]),
            percentile_stretch(image_chw[start + 1]),
            percentile_stretch(image_chw[start + 0]),
        ],
        axis=-1,
    )


def read_prob(path):
    with rasterio.open(path) as src:
        prob = src.read(1).astype(np.float32)
    if prob.max(initial=0) > 1.0:
        prob /= 255.0
    return prob


def binary_panel(mask, valid):
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    out[mask & valid] = (255, 255, 255)
    out[~valid] = (90, 90, 90)
    return out


def resize_panel(array, panel_size, nearest=False):
    method = Image.Resampling.NEAREST if nearest else Image.Resampling.LANCZOS
    return Image.fromarray(array).resize((panel_size, panel_size), method)


def load_font(size, bold=False):
    candidates = [
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/calibrib.ttf" if bold else "C:/Windows/Fonts/calibri.ttf"),
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def draw_panel_label(draw, x, y, label, font):
    bbox = draw.textbbox((0, 0), label, font=font)
    pad_x, pad_y = 12, 8
    draw.rounded_rectangle(
        (x, y, x + bbox[2] - bbox[0] + pad_x * 2, y + bbox[3] - bbox[1] + pad_y * 2),
        radius=4,
        fill=(255, 255, 255),
        outline=(25, 25, 25),
        width=2,
    )
    draw.text((x + pad_x, y + pad_y - 2), label, fill=(0, 0, 0), font=font)


def make_four_panel(rgb, gt, base_pred, finetune_pred, valid, panel_size, add_labels):
    panels = [
        resize_panel(rgb, panel_size),
        resize_panel(binary_panel(gt, valid), panel_size, nearest=True),
        resize_panel(binary_panel(base_pred, valid), panel_size, nearest=True),
        resize_panel(binary_panel(finetune_pred, valid), panel_size, nearest=True),
    ]
    gap = max(18, panel_size // 42)
    outer = max(14, panel_size // 64)
    width = outer * 2 + panel_size * 4 + gap * 3
    height = outer * 2 + panel_size
    canvas = Image.new("RGB", (width, height), "white")

    for i, panel in enumerate(panels):
        x = outer + i * (panel_size + gap)
        canvas.paste(panel, (x, outer))

    if add_labels:
        draw = ImageDraw.Draw(canvas)
        label_font = load_font(max(28, panel_size // 24), bold=True)
        for i, label in enumerate(["(A)", "(B)", "(C)", "(D)"]):
            x = outer + i * (panel_size + gap) + max(10, panel_size // 80)
            y = outer + max(10, panel_size // 80)
            draw_panel_label(draw, x, y, label, label_font)

    return canvas


def add_cm(total, cm):
    total += cm
    return total


def main():
    parser = argparse.ArgumentParser(
        description="Create Figure 4-1 style comparisons using the best adapted model outputs."
    )
    parser.add_argument(
        "--split",
        default="manual_tiles_filtered_problem_removed/filtered_trainval_split.json",
    )
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--base-prob-root", default="manual_tile_prediction_tifs_all")
    parser.add_argument("--finetune-prob-root", default="frequency_boundary_prediction_tifs_all")
    parser.add_argument("--output-dir", default="paper_figures/fig4_1_best_transfer_finetune_all")
    parser.add_argument("--base-threshold", type=float, default=0.12)
    parser.add_argument("--finetune-threshold", type=float, default=0.60)
    parser.add_argument("--rgb-time", type=int, default=4, help="0-based period index; 4 means W5.")
    parser.add_argument("--panel-size", type=int, default=1024)
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--no-labels", action="store_true")
    args = parser.parse_args()

    split = json.loads(Path(args.split).read_text(encoding="utf-8"))
    samples = split["train"] + split["val"]
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    rows = []
    missing = []
    base_total = np.zeros((2, 2), dtype=np.int64)
    finetune_total = np.zeros((2, 2), dtype=np.int64)

    for item in tqdm(samples, desc="Figure 4-1 all samples"):
        region = item["region"]
        sample = item["sample"]
        stem = Path(sample).stem
        region_dir = Path(args.data_root) / region
        base_prob_path = Path(args.base_prob_root) / region / "prob" / f"{stem}_prob.tif"
        finetune_prob_path = Path(args.finetune_prob_root) / region / "prob" / f"{stem}_prob.tif"
        if not base_prob_path.exists() or not finetune_prob_path.exists():
            missing.append(
                {
                    "region": region,
                    "sample": sample,
                    "base_prob": str(base_prob_path),
                    "finetune_prob": str(finetune_prob_path),
                }
            )
            continue

        image, gt_label = load_sample(region_dir, sample, ignore_nodata=True)
        valid = gt_label != IGNORE_LABEL
        gt = gt_label == 1
        base_pred = (read_prob(base_prob_path) >= args.base_threshold) & valid
        finetune_pred = (read_prob(finetune_prob_path) >= args.finetune_threshold) & valid

        base_cm = confusion_matrix(base_pred.astype(np.uint8), gt_label)
        finetune_cm = confusion_matrix(finetune_pred.astype(np.uint8), gt_label)
        add_cm(base_total, base_cm)
        add_cm(finetune_total, finetune_cm)

        rgb = make_rgb(image, args.rgb_time)
        figure = make_four_panel(
            rgb,
            gt,
            base_pred,
            finetune_pred,
            valid,
            args.panel_size,
            add_labels=not args.no_labels,
        )
        out_dir = out_root / region
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{stem}_fig4_1_transfer_finetune.png"
        figure.save(out_path, dpi=(args.dpi, args.dpi), compress_level=4)

        base_metrics = metrics_from_cm(base_cm)
        finetune_metrics = metrics_from_cm(finetune_cm)
        rows.append(
            {
                "region": region,
                "sample": sample,
                "figure": str(out_path.resolve()),
                "base_threshold": args.base_threshold,
                "finetune_threshold": args.finetune_threshold,
                "base_oa": base_metrics["oa"],
                "base_miou": base_metrics["miou"],
                "base_target_iou": base_metrics["iou"][1],
                "base_f1": base_metrics["f1"][1],
                "finetune_oa": finetune_metrics["oa"],
                "finetune_miou": finetune_metrics["miou"],
                "finetune_target_iou": finetune_metrics["iou"][1],
                "finetune_f1": finetune_metrics["f1"][1],
            }
        )

    manifest = out_root / "fig4_1_manifest.csv"
    with manifest.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "samples": len(rows),
        "missing": missing,
        "panel_order": [
            "(A) Sentinel-2 imagery",
            "(B) manually interpreted reference samples",
            "(C) direct-transfer results of the base model",
            "(D) results after region-adaptive fine-tuning",
        ],
        "base_prob_root": args.base_prob_root,
        "finetune_prob_root": args.finetune_prob_root,
        "base_threshold": args.base_threshold,
        "finetune_threshold": args.finetune_threshold,
        "base_confusion_matrix": base_total.tolist(),
        "finetune_confusion_matrix": finetune_total.tolist(),
        "base_metrics": metrics_from_cm(base_total),
        "finetune_metrics": metrics_from_cm(finetune_total),
    }
    summary_path = out_root / "fig4_1_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[INFO] Created {len(rows)} figures in: {out_root.resolve()}")
    print(f"[INFO] Manifest: {manifest.resolve()}")
    print(f"[INFO] Summary: {summary_path.resolve()}")
    if missing:
        print(f"[WARN] Missing {len(missing)} samples; see summary JSON.")


if __name__ == "__main__":
    main()

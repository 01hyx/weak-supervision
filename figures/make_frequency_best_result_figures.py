"""生成原始 RS3Mamba 与最佳小波频域微调结果的论文对比图。"""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import rasterio
import torch
from PIL import Image
from tqdm import tqdm

from evaluate_adaptation_experiments import build_experiment_model
from test_manual_tiles import IGNORE_LABEL, WINDOW_SIZE, confusion_matrix, load_sample, metrics_from_cm
from threshold_sweep_manual_tiles import infer_one_image_target_prob


def stretch(channel):
    valid = channel[np.isfinite(channel) & (channel > 0)]
    if not valid.size:
        return np.zeros_like(channel, dtype=np.uint8)
    lo, hi = np.percentile(valid, [2, 98])
    return (np.clip((channel - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)


def make_rgb(image, time_index=4):
    start = time_index * 6
    return np.stack(
        [stretch(image[start + 2]), stretch(image[start + 1]), stretch(image[start])],
        axis=-1,
    )


def read_prob(path):
    with rasterio.open(path) as src:
        prob = src.read(1).astype(np.float32)
    if prob.max(initial=0) > 1:
        prob /= 255.0
    return prob


def mask_panel(mask, valid):
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    out[mask & valid] = 255
    out[~valid] = 70
    return out


def boundary(mask, valid, width=2):
    mask = (mask & valid).astype(np.uint8)
    kernel = np.ones((width * 2 + 1, width * 2 + 1), dtype=np.uint8)
    return (cv2.dilate(mask, kernel) != cv2.erode(mask, kernel)) & valid


def boundary_comparison(rgb, gt, base, frequency, valid):
    out = rgb.copy()
    out[boundary(base, valid)] = (210, 0, 170)
    out[boundary(frequency, valid)] = (0, 210, 255)
    out[boundary(gt, valid)] = (255, 220, 0)
    out[~valid] = 65
    return out


def resize(array, size, nearest=False):
    resampling = Image.Resampling.NEAREST if nearest else Image.Resampling.LANCZOS
    return Image.fromarray(array).resize((size, size), resampling)


def make_figure(rgb, gt, base, frequency, valid, panel_size):
    panels = [
        resize(rgb, panel_size),
        resize(mask_panel(gt, valid), panel_size, True),
        resize(mask_panel(base, valid), panel_size, True),
        resize(mask_panel(frequency, valid), panel_size, True),
        resize(boundary_comparison(rgb, gt, base, frequency, valid), panel_size),
    ]
    gap = max(12, panel_size // 48)
    outer = max(10, panel_size // 64)
    canvas = Image.new(
        "RGB",
        (outer * 2 + panel_size * 5 + gap * 4, outer * 2 + panel_size),
        "white",
    )
    for index, panel in enumerate(panels):
        canvas.paste(panel, (outer + index * (panel_size + gap), outer))
    return canvas


def make_contact_sheet(paths, output_path, width=1920):
    images = [Image.open(path).convert("RGB") for path in paths]
    resized = []
    for image in images:
        height = round(width * image.height / image.width)
        resized.append(image.resize((width, height), Image.Resampling.LANCZOS))
    gap = 16
    canvas = Image.new("RGB", (width, sum(i.height for i in resized) + gap * (len(resized) - 1)), "white")
    y = 0
    for image in resized:
        canvas.paste(image, (0, y))
        y += image.height + gap
    canvas.save(output_path, dpi=(300, 300), compress_level=4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="manual_tiles_filtered_problem_removed/filtered_trainval_split.json")
    parser.add_argument("--split-part", choices=["train", "val"], default="val")
    parser.add_argument("--data-root", default=r"D:\s2_output\manual_tiles_maize30_stride128")
    parser.add_argument("--base-prob-root", default="manual_tile_prediction_tifs_all")
    parser.add_argument("--frequency-config", default="configs/experiment_frequency_wavelet.json")
    parser.add_argument("--output-dir", default="frequency_best_result_figures")
    parser.add_argument("--base-threshold", type=float, default=0.12)
    parser.add_argument("--frequency-threshold", type=float, default=0.60)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--panel-size", type=int, default=1024)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    config = json.loads(Path(args.frequency_config).read_text(encoding="utf-8"))
    model = build_experiment_model(config, device)
    samples = json.loads(Path(args.split).read_text(encoding="utf-8"))[args.split_part]
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    rows = []

    for item in tqdm(samples, desc="frequency result figures"):
        region, sample = item["region"], item["sample"]
        stem = Path(sample).stem
        image, label = load_sample(Path(args.data_root) / region, sample, ignore_nodata=True)
        valid = label != IGNORE_LABEL
        gt = label == 1
        base_prob = read_prob(Path(args.base_prob_root) / region / "prob" / f"{stem}_prob.tif")
        frequency_prob = infer_one_image_target_prob(
            model, image, device, args.stride, 1, WINDOW_SIZE
        )
        base = (base_prob >= args.base_threshold) & valid
        frequency = (frequency_prob >= args.frequency_threshold) & valid
        base_metrics = metrics_from_cm(confusion_matrix(base.astype(np.uint8), label))
        frequency_metrics = metrics_from_cm(confusion_matrix(frequency.astype(np.uint8), label))

        region_out = output_root / region
        region_out.mkdir(parents=True, exist_ok=True)
        output_path = region_out / f"{stem}_frequency_best_compare.png"
        make_figure(make_rgb(image), gt, base, frequency, valid, args.panel_size).save(
            output_path, dpi=(300, 300), compress_level=4
        )
        rows.append(
            {
                "region": region,
                "sample": sample,
                "figure": str(output_path.resolve()),
                "base_miou": base_metrics["miou"],
                "frequency_miou": frequency_metrics["miou"],
                "miou_improvement": frequency_metrics["miou"] - base_metrics["miou"],
                "base_target_iou": base_metrics["iou"][1],
                "frequency_target_iou": frequency_metrics["iou"][1],
                "target_iou_improvement": frequency_metrics["iou"][1] - base_metrics["iou"][1],
            }
        )

    rows.sort(key=lambda row: row["miou_improvement"], reverse=True)
    with (output_root / "frequency_best_result_manifest.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    make_contact_sheet(
        [Path(row["figure"]) for row in rows[:6]],
        output_root / "top_6_improvements_contact_sheet.png",
    )
    print(f"[INFO] Created {len(rows)} figures: {output_root.resolve()}")


if __name__ == "__main__":
    main()

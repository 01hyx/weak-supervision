"""频域边界引导的预测概率各向异性扩散后处理。"""

import argparse
import csv
import json
from itertools import product
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from eval_boundary_carve_postprocess import stretch_rgb_u8
from eval_wavelet_result_postprocess import spatial_frequency_score
from test_manual_tiles import DEFAULT_DATA_ROOT, IGNORE_LABEL, confusion_matrix, load_sample, metrics_from_cm
from tif_binary_postprocess import read_single_band, write_single_band


def shift(array, dy, dx):
    return np.roll(np.roll(array, dy, axis=0), dx, axis=1)


def frequency_guided_diffusion(prob, edge, iterations, alpha, edge_scale):
    current = prob.astype(np.float32).copy()
    seed_high = prob >= 0.88
    seed_low = prob <= 0.12
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for _ in range(iterations):
        weighted_sum = np.zeros_like(current)
        weight_sum = np.zeros_like(current)
        for dy, dx in directions:
            neighbor = shift(current, dy, dx)
            crossing_edge = np.maximum(edge, shift(edge, dy, dx))
            weight = np.exp(-crossing_edge / edge_scale).astype(np.float32)
            weighted_sum += neighbor * weight
            weight_sum += weight
        smooth = weighted_sum / np.maximum(weight_sum, 1e-6)
        current = (1.0 - alpha) * current + alpha * smooth
        current[seed_high] = prob[seed_high]
        current[seed_low] = prob[seed_low]
    return current


def eval_pred(pred, gt):
    out = np.full(gt.shape, IGNORE_LABEL, dtype=np.uint8)
    valid = gt != IGNORE_LABEL
    out[valid] = pred[valid].astype(np.uint8)
    return confusion_matrix(out, gt)


def save_figure(path, image, gt, before, after):
    valid = gt != IGNORE_LABEL
    panels = [stretch_rgb_u8(image)]
    for mask in (gt == 1, before, after):
        panel = np.zeros((*gt.shape, 3), dtype=np.uint8)
        panel[mask & valid] = 255
        panel[~valid] = 70
        panels.append(panel)
    change = stretch_rgb_u8(image)
    change[before & ~after & valid] = (230, 0, 160)
    change[~before & after & valid] = (0, 210, 255)
    panels.append(change)
    size, gap = 768, 14
    canvas = Image.new("RGB", (size * 5 + gap * 4, size), "white")
    for index, panel in enumerate(panels):
        canvas.paste(Image.fromarray(panel).resize((size, size), Image.Resampling.NEAREST), (index * (size + gap), 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, dpi=(300, 300), compress_level=4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-file", default="manual_tiles_filtered_problem_removed/filtered_trainval_split.json")
    parser.add_argument("--split-part", choices=["train", "val"], default="val")
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--pred-root", default="frequency_best_prediction_tifs_val")
    parser.add_argument("--output-dir", default="frequency_guided_diffusion")
    parser.add_argument("--save-best", action="store_true")
    args = parser.parse_args()

    items = json.loads(Path(args.split_file).read_text(encoding="utf-8"))[args.split_part]
    cases = []
    for item in items:
        region, sample = item["region"], item["sample"]
        image, gt = load_sample(Path(args.data_root) / region, sample, ignore_nodata=True)
        prob, profile = read_single_band(Path(args.pred_root) / region / "prob" / f"{Path(sample).stem}_prob.tif")
        cases.append((region, sample, image, prob.astype(np.float32), gt, profile, spatial_frequency_score(image)))

    cache = {}
    rows = []
    for iterations, alpha, edge_scale, threshold in product(
        [4, 8, 12, 20], [0.25, 0.45, 0.65], [0.06, 0.10, 0.16, 0.25], [0.56, 0.58, 0.60, 0.62]
    ):
        cm = np.zeros((2, 2), dtype=np.int64)
        changed = 0
        for index, (_, _, _, prob, gt, _, edge) in enumerate(cases):
            key = (index, iterations, alpha, edge_scale)
            if key not in cache:
                cache[key] = frequency_guided_diffusion(prob, edge, iterations, alpha, edge_scale)
            pred = cache[key] >= threshold
            cm += eval_pred(pred, gt)
            changed += int(np.count_nonzero(pred != (prob >= 0.60)))
        metrics = metrics_from_cm(cm)
        rows.append({
            "iterations": iterations, "alpha": alpha, "edge_scale": edge_scale, "threshold": threshold,
            "changed_pixels": changed, "oa": metrics["oa"], "f1": metrics["f1"][1],
            "target_iou": metrics["iou"][1], "miou": metrics["miou"],
            "fp": int(cm[0, 1]), "fn": int(cm[1, 0]),
        })
    rows.sort(key=lambda row: row["miou"], reverse=True)
    best = rows[0]
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with (output / f"{args.split_part}_diffusion_sweep.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    (output / f"{args.split_part}_diffusion_best.json").write_text(
        json.dumps(best, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if args.save_best:
        for region, sample, image, prob, gt, profile, edge in cases:
            refined_prob = frequency_guided_diffusion(
                prob, edge, int(best["iterations"]), best["alpha"], best["edge_scale"]
            )
            before = prob >= 0.60
            after = refined_prob >= best["threshold"]
            stem = Path(sample).stem
            valid = gt != IGNORE_LABEL
            out = np.full(gt.shape, IGNORE_LABEL, dtype=np.uint8)
            out[valid] = after[valid].astype(np.uint8)
            write_single_band(output / region / "pred" / f"{stem}_pred.tif", out, profile)
            write_single_band(output / region / "prob" / f"{stem}_prob.tif", refined_prob, profile, dtype="float32", nodata=None)
            save_figure(output / region / "figures" / f"{stem}_diffusion_compare.png", image, gt, before, after)
    print(json.dumps(best, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

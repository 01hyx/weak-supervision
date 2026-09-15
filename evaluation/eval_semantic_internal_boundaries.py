"""使用候选线两侧的多时相光谱差异筛选内部频域边界。"""

import argparse
import csv
import json
from itertools import product
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from eval_boundary_carve_postprocess import stretch_rgb_u8
from eval_internal_frequency_boundaries import (
    eval_pred,
    large_object_interior,
    temporal_consistent_edges,
)
from test_manual_tiles import DEFAULT_DATA_ROOT, IGNORE_LABEL, load_sample, metrics_from_cm
from tif_binary_postprocess import read_single_band, write_single_band


def shift(array, dx, dy):
    """平移数组并将越界位置设为 NaN，避免瓦片两侧循环污染。"""
    out = np.roll(array, shift=(dy, dx), axis=(-2, -1)).astype(np.float32, copy=False)
    if dy > 0:
        out[..., :dy, :] = np.nan
    elif dy < 0:
        out[..., dy:, :] = np.nan
    if dx > 0:
        out[..., :, :dx] = np.nan
    elif dx < 0:
        out[..., :, dx:] = np.nan
    return out


def spectral_boundary_features(image, prob, lines, distance):
    """计算候选线两侧差异、中心植被低谷和两侧预测支撑。"""
    data = image.reshape(6, 6, image.shape[-2], image.shape[-1]).astype(np.float32)
    periods = data[2:5]
    red = periods[:, 2]
    nir = periods[:, 3]
    ndvi = (nir - red) / np.maximum(nir + red, 1e-6)
    brightness = np.mean(periods[:, :4], axis=1)

    # 候选频域线的梯度方向近似为边界法向，按主方向选择两侧采样带。
    line_float = cv2.GaussianBlur(lines.astype(np.float32), (0, 0), 1.2)
    gx = cv2.Scharr(line_float, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(line_float, cv2.CV_32F, 0, 1)
    use_x = np.abs(gx) >= np.abs(gy)

    nx_pos, nx_neg = shift(ndvi, distance, 0), shift(ndvi, -distance, 0)
    ny_pos, ny_neg = shift(ndvi, 0, distance), shift(ndvi, 0, -distance)
    bx_pos, bx_neg = shift(brightness, distance, 0), shift(brightness, -distance, 0)
    by_pos, by_neg = shift(brightness, 0, distance), shift(brightness, 0, -distance)

    ndvi_pos = np.where(use_x[None], nx_pos, ny_pos)
    ndvi_neg = np.where(use_x[None], nx_neg, ny_neg)
    bright_pos = np.where(use_x[None], bx_pos, by_pos)
    bright_neg = np.where(use_x[None], bx_neg, by_neg)

    side_ndvi_diff = np.nanmean(np.abs(ndvi_pos - ndvi_neg), axis=0)
    bright_scale = np.nanpercentile(brightness, 95) - np.nanpercentile(brightness, 5) + 1e-6
    side_brightness_diff = np.nanmean(np.abs(bright_pos - bright_neg), axis=0) / bright_scale
    side_mean_ndvi = np.nanmean(0.5 * (ndvi_pos + ndvi_neg), axis=0)
    center_ndvi = np.nanmean(ndvi, axis=0)
    center_gap = np.maximum(side_mean_ndvi - center_ndvi, 0.0)

    prob_pos_x, prob_neg_x = shift(prob, distance, 0), shift(prob, -distance, 0)
    prob_pos_y, prob_neg_y = shift(prob, 0, distance), shift(prob, 0, -distance)
    prob_pos = np.where(use_x, prob_pos_x, prob_pos_y)
    prob_neg = np.where(use_x, prob_neg_x, prob_neg_y)
    side_support = np.minimum(prob_pos, prob_neg)
    probability_valley = np.maximum(0.5 * (prob_pos + prob_neg) - prob, 0.0)

    for feature in (side_ndvi_diff, side_brightness_diff, center_gap, side_support, probability_valley):
        feature[~np.isfinite(feature)] = 0
    return side_ndvi_diff, side_brightness_diff, center_gap, side_support, probability_valley


def robust_unit(feature, mask):
    values = feature[mask]
    if not values.size:
        return np.zeros_like(feature, dtype=np.float32)
    low, high = np.quantile(values, [0.10, 0.95])
    return np.clip((feature - low) / max(float(high - low), 1e-6), 0, 1).astype(np.float32)


def prepare_case(image, prob):
    initial = prob >= 0.60
    interior = large_object_interior(initial, min_object_area=2500, margin=1)
    lines, _ = temporal_consistent_edges(image, interior, quantile=0.78, min_votes=2, min_length=12)
    features = spectral_boundary_features(image, prob, lines, distance=2)
    side_ndvi, side_brightness, center_gap, side_support, probability_valley = features
    score = (
        0.34 * robust_unit(side_ndvi, lines)
        + 0.16 * robust_unit(side_brightness, lines)
        + 0.30 * robust_unit(center_gap, lines)
        + 0.20 * robust_unit(probability_valley, lines)
    )
    return initial, interior, lines, score, side_support, center_gap


def refine(prepared, prob, score_quantile, carve_prob, side_support_min, gap_min):
    initial, _, lines, score, side_support, center_gap = prepared
    values = score[lines]
    threshold = float(np.quantile(values, score_quantile)) if values.size else 1.0
    # 光谱证据强时可切开较高置信预测；中心植被低谷则作为道路/沟渠的补充证据。
    accepted = lines & (score >= threshold) & (side_support >= side_support_min)
    accepted &= (prob < carve_prob) | (center_gap >= gap_min)
    refined = initial.copy()
    refined[accepted] = False
    return refined, accepted, threshold


def save_figure(path, image, gt, before, after, prepared, accepted):
    valid = gt != IGNORE_LABEL
    _, interior, lines, score, _, _ = prepared
    rgb = stretch_rgb_u8(image)
    panels = [rgb]
    for mask in (gt == 1, before, after):
        panel = np.zeros((*gt.shape, 3), dtype=np.uint8)
        panel[mask & valid] = 255
        panel[~valid] = 70
        panels.append(panel)
    debug = rgb.copy()
    debug[interior & valid] = (0, 120, 205)
    debug[lines & valid] = (255, 220, 0)
    strong = lines & (score >= np.quantile(score[lines], 0.70) if lines.any() else False)
    debug[strong & valid] = (255, 120, 0)
    debug[accepted & valid] = (230, 0, 160)
    panels.append(debug)
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
    parser.add_argument("--output-dir", default="semantic_internal_boundaries")
    parser.add_argument("--save-best", action="store_true")
    args = parser.parse_args()

    items = json.loads(Path(args.split_file).read_text(encoding="utf-8"))[args.split_part]
    cases = []
    for item in items:
        region, sample = item["region"], item["sample"]
        image, gt = load_sample(Path(args.data_root) / region, sample, ignore_nodata=True)
        prob, profile = read_single_band(Path(args.pred_root) / region / "prob" / f"{Path(sample).stem}_prob.tif")
        prob = prob.astype(np.float32)
        cases.append((region, sample, image, prob, gt, profile, prepare_case(image, prob)))

    baseline_cm = np.zeros((2, 2), dtype=np.int64)
    for _, _, _, prob, gt, _, _ in cases:
        baseline_cm += eval_pred(prob >= 0.60, gt)
    baseline = metrics_from_cm(baseline_cm)

    rows = []
    search = product([0.45, 0.55, 0.65, 0.72, 0.80], [0.68, 0.74, 0.80, 0.86, 0.92], [0.50, 0.58, 0.64], [0.025, 0.05, 0.08])
    for score_quantile, carve_prob, side_support_min, gap_min in search:
        cm = np.zeros((2, 2), dtype=np.int64)
        carved_pixels = 0
        for _, _, _, prob, gt, _, prepared in cases:
            refined, accepted, _ = refine(prepared, prob, score_quantile, carve_prob, side_support_min, gap_min)
            cm += eval_pred(refined, gt)
            carved_pixels += int(accepted.sum())
        metrics = metrics_from_cm(cm)
        rows.append({
            "score_quantile": score_quantile, "carve_prob": carve_prob, "side_support_min": side_support_min,
            "gap_min": gap_min, "carved_pixels": carved_pixels, "oa": metrics["oa"], "f1": metrics["f1"][1],
            "target_iou": metrics["iou"][1], "miou": metrics["miou"], "fp": int(cm[0, 1]), "fn": int(cm[1, 0]),
        })
    rows.sort(key=lambda row: row["miou"], reverse=True)
    best = rows[0]
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with (output / f"{args.split_part}_semantic_internal_sweep.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    if args.save_best:
        for region, sample, image, prob, gt, profile, prepared in cases:
            refined, accepted, _ = refine(
                prepared, prob, best["score_quantile"], best["carve_prob"], best["side_support_min"], best["gap_min"]
            )
            valid = gt != IGNORE_LABEL
            out = np.full(gt.shape, IGNORE_LABEL, dtype=np.uint8)
            out[valid] = refined[valid].astype(np.uint8)
            stem = Path(sample).stem
            write_single_band(output / region / "pred" / f"{stem}_pred.tif", out, profile)
            save_figure(output / region / "figures" / f"{stem}_semantic_internal_compare.png", image, gt, prob >= 0.60, refined, prepared, accepted)

    summary = {"baseline": baseline, "best": best}
    (output / f"{args.split_part}_semantic_internal_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

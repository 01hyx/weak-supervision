"""仅在预测边界缓冲区内使用 W3-W5 分位数频域边界修正 Mask。"""

import argparse
import csv
import json
from itertools import product
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from eval_boundary_carve_postprocess import stretch_rgb_u8
from eval_wavelet_result_postprocess import haar_detail, normalize_robust
from test_manual_tiles import DEFAULT_DATA_ROOT, IGNORE_LABEL, confusion_matrix, load_sample, metrics_from_cm
from tif_binary_postprocess import read_single_band, write_single_band


def prediction_boundary_buffer(mask, radius):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    dilated = cv2.dilate(mask.astype(np.uint8), kernel) > 0
    eroded = cv2.erode(mask.astype(np.uint8), kernel) > 0
    return dilated & ~eroded


def period_frequency_score(data, period_index):
    red = data[period_index, 2]
    nir = data[period_index, 3]
    ndvi = (nir - red) / np.maximum(nir + red, 1e-6)
    brightness = np.mean(data[period_index, :4], axis=0)
    haar = 0.40 * normalize_robust(haar_detail(brightness)) + 0.45 * normalize_robust(haar_detail(ndvi))

    # 多尺度 Scharr 梯度用于补充 Haar 容易漏掉的弱对比、连续田块边界。
    gradient_scores = []
    for source in (normalize_robust(brightness), normalize_robust(ndvi)):
        for sigma in (0.8, 1.6, 2.4):
            smooth = cv2.GaussianBlur(source, (0, 0), sigma)
            gx = cv2.Scharr(smooth, cv2.CV_32F, 1, 0)
            gy = cv2.Scharr(smooth, cv2.CV_32F, 0, 1)
            gradient_scores.append(normalize_robust(cv2.magnitude(gx, gy)))
    gradient = np.max(np.stack(gradient_scores, axis=0), axis=0)
    return normalize_robust(haar + 0.35 * gradient)


def focused_frequency_score(image):
    data = image.reshape(6, 6, image.shape[-2], image.shape[-1])
    scores = np.stack([period_frequency_score(data, index) for index in (2, 3, 4)], axis=0)
    # 中位数保留多期稳定边界，最大值补回仅在某一期清晰的边界。
    stable = np.median(scores, axis=0)
    temporal_best = np.max(scores, axis=0)
    score = normalize_robust(0.55 * stable + 0.45 * temporal_best)
    return cv2.GaussianBlur(score.astype(np.float32), (3, 3), 0)


def refine(prob, score, threshold, buffer_radius, quantile, carve_prob, add_prob, line_width):
    initial = prob >= threshold
    buffer = prediction_boundary_buffer(initial, buffer_radius)
    values = score[buffer]
    edge_threshold = float(np.quantile(values, quantile)) if values.size else 1.0
    strong_edge = buffer & (score >= edge_threshold)
    # 连接短小断裂边界，但闭运算仍限制在模型边界缓冲区内。
    horizontal = cv2.morphologyEx(
        strong_edge.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((1, 5), dtype=np.uint8)
    )
    vertical = cv2.morphologyEx(
        strong_edge.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 1), dtype=np.uint8)
    )
    strong_edge = ((horizontal > 0) | (vertical > 0)) & buffer
    if line_width > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (line_width, line_width))
        strong_edge = cv2.dilate(strong_edge.astype(np.uint8), kernel) > 0
        strong_edge &= buffer
    carved = initial & strong_edge & (prob < carve_prob)
    add = (~initial) & buffer & (~strong_edge) & (prob >= add_prob)
    refined = initial.copy()
    refined[carved] = False
    refined[add] = True
    return refined, buffer, strong_edge, carved, add, edge_threshold


def eval_pred(pred, gt):
    out = np.full(gt.shape, IGNORE_LABEL, dtype=np.uint8)
    valid = gt != IGNORE_LABEL
    out[valid] = pred[valid].astype(np.uint8)
    return confusion_matrix(out, gt)


def save_figure(path, image, gt, before, after, buffer, edge, carved, added):
    valid = gt != IGNORE_LABEL
    rgb = stretch_rgb_u8(image)
    panels = [rgb]
    for mask in (gt == 1, before, after):
        panel = np.zeros((*gt.shape, 3), dtype=np.uint8)
        panel[mask & valid] = 255
        panel[~valid] = 70
        panels.append(panel)
    debug = rgb.copy()
    debug[buffer & valid] = (0, 170, 255)
    debug[edge & valid] = (255, 220, 0)
    debug[carved & valid] = (230, 0, 160)
    debug[added & valid] = (0, 255, 255)
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
    parser.add_argument("--output-dir", default="local_quantile_frequency_boundary")
    parser.add_argument("--save-best", action="store_true")
    args = parser.parse_args()

    items = json.loads(Path(args.split_file).read_text(encoding="utf-8"))[args.split_part]
    cases = []
    for item in items:
        region, sample = item["region"], item["sample"]
        image, gt = load_sample(Path(args.data_root) / region, sample, ignore_nodata=True)
        prob, profile = read_single_band(Path(args.pred_root) / region / "prob" / f"{Path(sample).stem}_prob.tif")
        cases.append((region, sample, image, prob.astype(np.float32), gt, profile, focused_frequency_score(image)))

    baseline_cm = np.zeros((2, 2), dtype=np.int64)
    for _, _, _, prob, gt, _, _ in cases:
        baseline_cm += eval_pred(prob >= 0.60, gt)
    baseline = metrics_from_cm(baseline_cm)

    rows = []
    for buffer_radius, quantile, carve_prob, add_prob, line_width in product(
        [2, 3, 4, 5], [0.70, 0.80, 0.85, 0.90], [0.65, 0.68, 0.72], [0.52, 0.55, 0.58], [1, 2]
    ):
        cm = np.zeros((2, 2), dtype=np.int64)
        carved_pixels = 0
        for _, _, _, prob, gt, _, score in cases:
            refined, _, _, carved, added, _ = refine(prob, score, 0.60, buffer_radius, quantile, carve_prob, add_prob, line_width)
            cm += eval_pred(refined, gt)
            carved_pixels += int(carved.sum())
        metrics = metrics_from_cm(cm)
        rows.append({
            "buffer_radius": buffer_radius, "quantile": quantile, "carve_prob": carve_prob,
            "add_prob": add_prob, "line_width": line_width, "carved_pixels": carved_pixels, "oa": metrics["oa"],
            "f1": metrics["f1"][1], "target_iou": metrics["iou"][1], "miou": metrics["miou"],
            "fp": int(cm[0, 1]), "fn": int(cm[1, 0]),
        })
    rows.sort(key=lambda row: row["miou"], reverse=True)
    best = rows[0]
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with (output / f"{args.split_part}_local_quantile_sweep.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    if args.save_best:
        for region, sample, image, prob, gt, profile, score in cases:
            refined, buffer, edge, carved, added, _ = refine(
                prob, score, 0.60, int(best["buffer_radius"]), best["quantile"], best["carve_prob"], best["add_prob"], int(best["line_width"])
            )
            valid = gt != IGNORE_LABEL
            out = np.full(gt.shape, IGNORE_LABEL, dtype=np.uint8)
            out[valid] = refined[valid].astype(np.uint8)
            stem = Path(sample).stem
            write_single_band(output / region / "pred" / f"{stem}_pred.tif", out, profile)
            save_figure(output / region / "figures" / f"{stem}_local_frequency_compare.png", image, gt, prob >= 0.60, refined, buffer, edge, carved, added)

    summary = {"baseline": baseline, "best": best}
    (output / f"{args.split_part}_local_quantile_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

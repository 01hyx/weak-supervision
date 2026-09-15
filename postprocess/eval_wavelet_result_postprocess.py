"""使用六期影像 Haar 高频信息对预测结果进行空间-频域后处理。"""

import argparse
import csv
import json
from itertools import product
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from eval_boundary_carve_postprocess import stretch_rgb_u8
from test_manual_tiles import DEFAULT_DATA_ROOT, IGNORE_LABEL, confusion_matrix, load_sample, metrics_from_cm
from tif_binary_postprocess import read_single_band, remove_small_components, write_single_band


def normalize_robust(array):
    valid = array[np.isfinite(array)]
    if not valid.size:
        return np.zeros_like(array, dtype=np.float32)
    lo, hi = np.percentile(valid, [5, 95])
    return np.clip((array - lo) / max(hi - lo, 1e-6), 0, 1).astype(np.float32)


def haar_detail(array):
    h, w = array.shape
    if h % 2 or w % 2:
        array = cv2.copyMakeBorder(array, 0, h % 2, 0, w % 2, cv2.BORDER_REPLICATE)
    x00 = array[0::2, 0::2]
    x01 = array[0::2, 1::2]
    x10 = array[1::2, 0::2]
    x11 = array[1::2, 1::2]
    lh = np.abs(-x00 - x01 + x10 + x11)
    hl = np.abs(-x00 + x01 - x10 + x11)
    hh = np.abs(x00 - x01 - x10 + x11)
    detail = np.sqrt(lh * lh + hl * hl + hh * hh)
    return cv2.resize(detail, (w, h), interpolation=cv2.INTER_LINEAR)


def spatial_frequency_score(image):
    data = image.reshape(6, 6, image.shape[-2], image.shape[-1])
    scores = []
    for time_index in range(6):
        red = data[time_index, 2]
        nir = data[time_index, 3]
        ndvi = (nir - red) / np.maximum(nir + red, 1e-6)
        brightness = np.mean(data[time_index, :4], axis=0)
        scores.append(normalize_robust(haar_detail(ndvi)))
        scores.append(normalize_robust(haar_detail(brightness)))
    # 多时相中重复出现的高频边界更可靠，使用中位数抑制单期噪声。
    score = np.median(np.stack(scores, axis=0), axis=0)
    return cv2.GaussianBlur(score, (3, 3), 0)


def refine_result(prob, score, threshold, edge_threshold, remove_prob, add_prob, radius, min_area):
    pred = prob >= threshold
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    dilated = cv2.dilate(pred.astype(np.uint8), kernel) > 0
    eroded = cv2.erode(pred.astype(np.uint8), kernel) > 0
    inner_boundary = pred & ~eroded
    outer_boundary = ~pred & dilated

    # 高频强、置信度不足的边界像元倾向于道路、沟渠或地块间隙。
    remove = inner_boundary & (score >= edge_threshold) & (prob < remove_prob)
    # 高频弱且接近分类阈值的邻接像元倾向于同一连片地块中的漏检。
    add = outer_boundary & (score < edge_threshold * 0.75) & (prob >= add_prob)
    refined = pred.copy()
    refined[remove] = False
    refined[add] = True
    refined = remove_small_components(refined, min_area)
    return refined, remove, add


def eval_pred(pred, gt):
    out = np.full(gt.shape, IGNORE_LABEL, dtype=np.uint8)
    valid = gt != IGNORE_LABEL
    out[valid] = pred[valid].astype(np.uint8)
    return confusion_matrix(out, gt)


def save_figure(path, image, gt, before, after, remove, add):
    rgb = stretch_rgb_u8(image)
    valid = gt != IGNORE_LABEL
    panels = [rgb]
    for mask in (gt == 1, before, after):
        panel = np.zeros((*gt.shape, 3), dtype=np.uint8)
        panel[mask & valid] = 255
        panel[~valid] = 70
        panels.append(panel)
    change = rgb.copy()
    change[remove & valid] = (230, 0, 160)
    change[add & valid] = (0, 210, 255)
    panels.append(change)
    size = 768
    gap = 14
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
    parser.add_argument("--output-dir", default="frequency_wavelet_result_postprocess")
    parser.add_argument("--save-best", action="store_true")
    parser.add_argument("--refine-search", action="store_true")
    args = parser.parse_args()

    items = json.loads(Path(args.split_file).read_text(encoding="utf-8"))[args.split_part]
    cases = []
    for item in items:
        region, sample = item["region"], item["sample"]
        image, gt = load_sample(Path(args.data_root) / region, sample, ignore_nodata=True)
        prob, profile = read_single_band(Path(args.pred_root) / region / "prob" / f"{Path(sample).stem}_prob.tif")
        cases.append((region, sample, image, prob.astype(np.float32), gt, profile, spatial_frequency_score(image)))

    baseline_cm = np.zeros((2, 2), dtype=np.int64)
    for _, _, _, prob, gt, _, _ in cases:
        baseline_cm += eval_pred(prob >= 0.60, gt)
    baseline = metrics_from_cm(baseline_cm)

    if args.refine_search:
        search_values = ([0.20, 0.25, 0.30, 0.35], [0.66, 0.68, 0.70], [0.50, 0.52, 0.54], [1, 2, 3], [0, 20])
    else:
        search_values = ([0.30, 0.40, 0.50, 0.60], [0.62, 0.68, 0.75], [0.48, 0.52, 0.56], [1, 2], [0, 20, 50])
    rows = []
    for edge_threshold, remove_prob, add_prob, radius, min_area in product(*search_values):
        cm = np.zeros((2, 2), dtype=np.int64)
        removed = added = 0
        for _, _, _, prob, gt, _, score in cases:
            refined, remove, add = refine_result(prob, score, 0.60, edge_threshold, remove_prob, add_prob, radius, min_area)
            cm += eval_pred(refined, gt)
            removed += int(remove.sum())
            added += int(add.sum())
        metrics = metrics_from_cm(cm)
        rows.append({
            "edge_threshold": edge_threshold, "remove_prob": remove_prob, "add_prob": add_prob,
            "radius": radius, "min_area": min_area, "removed_pixels": removed, "added_pixels": added,
            "oa": metrics["oa"], "f1": metrics["f1"][1], "target_iou": metrics["iou"][1],
            "miou": metrics["miou"], "fp": int(cm[0, 1]), "fn": int(cm[1, 0]),
        })
    rows.sort(key=lambda row: row["miou"], reverse=True)
    best = rows[0]
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with (output / f"{args.split_part}_wavelet_result_sweep.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    if args.save_best:
        for region, sample, image, prob, gt, profile, score in cases:
            refined, remove, add = refine_result(
                prob, score, 0.60, best["edge_threshold"], best["remove_prob"],
                best["add_prob"], int(best["radius"]), int(best["min_area"])
            )
            valid = gt != IGNORE_LABEL
            out = np.full(gt.shape, IGNORE_LABEL, dtype=np.uint8)
            out[valid] = refined[valid].astype(np.uint8)
            stem = Path(sample).stem
            write_single_band(output / region / "pred" / f"{stem}_pred.tif", out, profile)
            save_figure(output / region / "figures" / f"{stem}_wavelet_result_compare.png", image, gt, prob >= 0.60, refined, remove, add)

    summary = {"baseline": baseline, "best": best}
    (output / f"{args.split_part}_wavelet_result_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

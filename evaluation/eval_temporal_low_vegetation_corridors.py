"""提取 W3-W5 持续低植被线状走廊，用于修正大型预测区域内部边界。"""

import argparse
import csv
import json
from itertools import product
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from eval_boundary_carve_postprocess import stretch_rgb_u8
from eval_internal_frequency_boundaries import eval_pred, large_object_interior, remove_short_components
from test_manual_tiles import DEFAULT_DATA_ROOT, IGNORE_LABEL, load_sample, metrics_from_cm
from tif_binary_postprocess import read_single_band, write_single_band


def temporal_corridor_evidence(image, interior, quantile, min_votes, line_length, min_length):
    data = image.reshape(6, 6, image.shape[-2], image.shape[-1]).astype(np.float32)
    votes = np.zeros(interior.shape, dtype=np.uint8)
    scores = []
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    for period_index in (2, 3, 4):
        red = data[period_index, 2]
        nir = data[period_index, 3]
        ndvi = (nir - red) / np.maximum(nir + red, 1e-6)
        smooth = cv2.GaussianBlur(ndvi, (3, 3), 0)
        # 闭运算填平窄低值带，其与原值之差突出道路、沟渠和田块间隙。
        corridor_score = np.maximum(cv2.morphologyEx(smooth, cv2.MORPH_CLOSE, close_kernel) - smooth, 0)
        scores.append(corridor_score)
        values = corridor_score[interior]
        threshold = float(np.quantile(values, quantile)) if values.size else 1.0
        votes += (corridor_score >= threshold).astype(np.uint8)

    candidate = interior & (votes >= min_votes)
    horizontal = cv2.morphologyEx(
        candidate.astype(np.uint8), cv2.MORPH_OPEN, np.ones((1, line_length), dtype=np.uint8)
    )
    vertical = cv2.morphologyEx(
        candidate.astype(np.uint8), cv2.MORPH_OPEN, np.ones((line_length, 1), dtype=np.uint8)
    )
    lines = ((horizontal > 0) | (vertical > 0)) & interior
    lines = cv2.morphologyEx(lines.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)) > 0
    lines &= interior
    return remove_short_components(lines, min_length), votes, np.median(np.stack(scores), axis=0)


def prepare_case(image, prob, quantile, min_votes, line_length, min_length):
    initial = prob >= 0.60
    interior = large_object_interior(initial, min_object_area=1200, margin=1)
    corridors, votes, score = temporal_corridor_evidence(
        image, interior, quantile, min_votes, line_length, min_length
    )
    return initial, interior, corridors, votes, score


def refine(prepared, prob, carve_prob, dilate_width):
    initial, interior, corridors, votes, score = prepared
    if dilate_width > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_width, dilate_width))
        carve = cv2.dilate(corridors.astype(np.uint8), kernel) > 0
        carve &= interior
    else:
        carve = corridors.copy()
    carve &= prob < carve_prob
    refined = initial.copy()
    refined[carve] = False
    return refined, carve


def save_figure(path, image, gt, before, after, prepared, carved):
    valid = gt != IGNORE_LABEL
    _, interior, corridors, _, _ = prepared
    rgb = stretch_rgb_u8(image)
    panels = [rgb]
    for mask in (gt == 1, before, after):
        panel = np.zeros((*gt.shape, 3), dtype=np.uint8)
        panel[mask & valid] = 255
        panel[~valid] = 70
        panels.append(panel)
    debug = rgb.copy()
    debug[interior & valid] = (0, 120, 205)
    debug[corridors & valid] = (255, 210, 0)
    debug[carved & valid] = (230, 0, 160)
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
    parser.add_argument("--output-dir", default="temporal_low_vegetation_corridors")
    parser.add_argument("--save-best", action="store_true")
    parser.add_argument("--fast-search", action="store_true")
    args = parser.parse_args()

    items = json.loads(Path(args.split_file).read_text(encoding="utf-8"))[args.split_part]
    raw_cases = []
    for item in items:
        region, sample = item["region"], item["sample"]
        image, gt = load_sample(Path(args.data_root) / region, sample, ignore_nodata=True)
        prob, profile = read_single_band(Path(args.pred_root) / region / "prob" / f"{Path(sample).stem}_prob.tif")
        raw_cases.append((region, sample, image, prob.astype(np.float32), gt, profile))

    baseline_cm = np.zeros((2, 2), dtype=np.int64)
    for _, _, _, prob, gt, _ in raw_cases:
        baseline_cm += eval_pred(prob >= 0.60, gt)
    baseline = metrics_from_cm(baseline_cm)

    rows = []
    cache = {}
    if args.fast_search:
        corridor_search = product([0.78, 0.86], [2, 3], [5, 7], [8, 16])
        refine_search = list(product([0.72, 0.82, 0.90], [1, 3]))
    else:
        corridor_search = product([0.72, 0.78, 0.84, 0.90], [2, 3], [5, 7, 9], [8, 12, 20])
        refine_search = list(product([0.68, 0.74, 0.80, 0.86, 0.92], [1, 3]))
    for quantile, min_votes, line_length, min_length in corridor_search:
        key = (quantile, min_votes, line_length, min_length)
        cache[key] = [prepare_case(image, prob, *key) for _, _, image, prob, _, _ in raw_cases]
        for carve_prob, dilate_width in refine_search:
            cm = np.zeros((2, 2), dtype=np.int64)
            carved_pixels = 0
            for case, prepared in zip(raw_cases, cache[key]):
                prob, gt = case[3], case[4]
                refined, carved = refine(prepared, prob, carve_prob, dilate_width)
                cm += eval_pred(refined, gt)
                carved_pixels += int(carved.sum())
            metrics = metrics_from_cm(cm)
            rows.append({
                "quantile": quantile, "min_votes": min_votes, "line_length": line_length,
                "min_length": min_length, "carve_prob": carve_prob, "dilate_width": dilate_width,
                "carved_pixels": carved_pixels, "oa": metrics["oa"], "f1": metrics["f1"][1],
                "target_iou": metrics["iou"][1], "miou": metrics["miou"],
                "fp": int(cm[0, 1]), "fn": int(cm[1, 0]),
            })
    rows.sort(key=lambda row: row["miou"], reverse=True)
    best = rows[0]
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with (output / f"{args.split_part}_corridor_sweep.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    if args.save_best:
        key = (best["quantile"], int(best["min_votes"]), int(best["line_length"]), int(best["min_length"]))
        for case, prepared in zip(raw_cases, cache[key]):
            region, sample, image, prob, gt, profile = case
            refined, carved = refine(prepared, prob, best["carve_prob"], int(best["dilate_width"]))
            valid = gt != IGNORE_LABEL
            out = np.full(gt.shape, IGNORE_LABEL, dtype=np.uint8)
            out[valid] = refined[valid].astype(np.uint8)
            stem = Path(sample).stem
            write_single_band(output / region / "pred" / f"{stem}_pred.tif", out, profile)
            save_figure(output / region / "figures" / f"{stem}_corridor_compare.png", image, gt, prob >= 0.60, refined, prepared, carved)

    summary = {"baseline": baseline, "best": best}
    (output / f"{args.split_part}_corridor_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

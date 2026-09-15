"""在大型预测连通域内部提取多期一致的连续频域线并切分粘连地块。"""

import argparse
import csv
import json
from itertools import product
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from eval_boundary_carve_postprocess import stretch_rgb_u8
from eval_local_quantile_frequency_boundary import (
    focused_frequency_score,
    period_frequency_score,
    prediction_boundary_buffer,
)
from test_manual_tiles import DEFAULT_DATA_ROOT, IGNORE_LABEL, confusion_matrix, load_sample, metrics_from_cm
from tif_binary_postprocess import connected_components, read_single_band, write_single_band


def remove_short_components(binary, min_length):
    labels, components = connected_components(binary)
    out = np.zeros_like(binary, dtype=bool)
    for index, coords in enumerate(components, start=1):
        if coords.shape[0] < min_length:
            continue
        ys, xs = coords[:, 0], coords[:, 1]
        span = max(int(ys.max() - ys.min() + 1), int(xs.max() - xs.min() + 1))
        if span >= min_length:
            out[labels == index] = True
    return out


def large_object_interior(pred, min_object_area, margin):
    labels, components = connected_components(pred)
    interior = np.zeros_like(pred, dtype=bool)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (margin * 2 + 1, margin * 2 + 1))
    for index, coords in enumerate(components, start=1):
        if coords.shape[0] < min_object_area:
            continue
        obj = labels == index
        eroded = cv2.erode(obj.astype(np.uint8), kernel) > 0
        interior |= eroded
    return interior


def temporal_consistent_edges(image, interior, quantile, min_votes, min_length):
    data = image.reshape(6, 6, image.shape[-2], image.shape[-1])
    period_scores = [period_frequency_score(data, index) for index in (2, 3, 4)]
    votes = np.zeros(interior.shape, dtype=np.uint8)
    for score in period_scores:
        values = score[interior]
        threshold = float(np.quantile(values, quantile)) if values.size else 1.0
        votes += (score >= threshold).astype(np.uint8)
    candidate = interior & (votes >= min_votes)
    horizontal = cv2.morphologyEx(candidate.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((1, 7), np.uint8))
    vertical = cv2.morphologyEx(candidate.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((7, 1), np.uint8))
    connected = ((horizontal > 0) | (vertical > 0)) & interior
    return remove_short_components(connected, min_length), votes


def refine(prob, image, threshold, quantile, min_votes, min_length, min_object_area, margin, carve_prob):
    pred = prob >= threshold
    interior = large_object_interior(pred, min_object_area, margin)
    lines, votes = temporal_consistent_edges(image, interior, quantile, min_votes, min_length)
    carved = lines & (prob < carve_prob)
    refined = pred.copy()
    refined[carved] = False
    return refined, interior, lines, carved, votes


def eval_pred(pred, gt):
    out = np.full(gt.shape, IGNORE_LABEL, dtype=np.uint8)
    valid = gt != IGNORE_LABEL
    out[valid] = pred[valid].astype(np.uint8)
    return confusion_matrix(out, gt)


def save_figure(path, image, gt, before, after, interior, lines, carved):
    valid = gt != IGNORE_LABEL
    rgb = stretch_rgb_u8(image)
    panels = [rgb]
    for mask in (gt == 1, before, after):
        panel = np.zeros((*gt.shape, 3), dtype=np.uint8)
        panel[mask & valid] = 255
        panel[~valid] = 70
        panels.append(panel)
    debug = rgb.copy()
    debug[interior & valid] = (0, 125, 205)
    debug[lines & valid] = (255, 220, 0)
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
    parser.add_argument("--output-dir", default="internal_frequency_boundaries")
    parser.add_argument("--save-best", action="store_true")
    parser.add_argument("--fast-search", action="store_true")
    args = parser.parse_args()

    items = json.loads(Path(args.split_file).read_text(encoding="utf-8"))[args.split_part]
    cases = []
    for item in items:
        region, sample = item["region"], item["sample"]
        image, gt = load_sample(Path(args.data_root) / region, sample, ignore_nodata=True)
        prob, profile = read_single_band(Path(args.pred_root) / region / "prob" / f"{Path(sample).stem}_prob.tif")
        cases.append((region, sample, image, prob.astype(np.float32), gt, profile))

    baseline_cm = np.zeros((2, 2), dtype=np.int64)
    for _, _, _, prob, gt, _ in cases:
        baseline_cm += eval_pred(prob >= 0.60, gt)
    baseline = metrics_from_cm(baseline_cm)

    if args.fast_search:
        search_values = ([0.78, 0.85, 0.90], [2, 3], [12, 24, 36], [1200, 2500], [1, 2], [0.64, 0.68])
    else:
        search_values = ([0.75, 0.82, 0.88, 0.92], [2, 3], [8, 16, 24, 32], [800, 1600, 3000], [1, 2, 3], [0.62, 0.66, 0.70])
    rows = []
    for quantile, min_votes, min_length, min_object_area, margin, carve_prob in product(*search_values):
        cm = np.zeros((2, 2), dtype=np.int64)
        carved_pixels = 0
        for _, _, image, prob, gt, _ in cases:
            refined, _, _, carved, _ = refine(
                prob, image, 0.60, quantile, min_votes, min_length, min_object_area, margin, carve_prob
            )
            cm += eval_pred(refined, gt)
            carved_pixels += int(carved.sum())
        metrics = metrics_from_cm(cm)
        rows.append({
            "quantile": quantile, "min_votes": min_votes, "min_length": min_length,
            "min_object_area": min_object_area, "margin": margin, "carve_prob": carve_prob,
            "carved_pixels": carved_pixels, "oa": metrics["oa"], "f1": metrics["f1"][1],
            "target_iou": metrics["iou"][1], "miou": metrics["miou"],
            "fp": int(cm[0, 1]), "fn": int(cm[1, 0]),
        })
    rows.sort(key=lambda row: row["miou"], reverse=True)
    best = rows[0]
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with (output / f"{args.split_part}_internal_frequency_sweep.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    if args.save_best:
        for region, sample, image, prob, gt, profile in cases:
            refined, interior, lines, carved, _ = refine(
                prob, image, 0.60, best["quantile"], int(best["min_votes"]), int(best["min_length"]),
                int(best["min_object_area"]), int(best["margin"]), best["carve_prob"]
            )
            valid = gt != IGNORE_LABEL
            out = np.full(gt.shape, IGNORE_LABEL, dtype=np.uint8)
            out[valid] = refined[valid].astype(np.uint8)
            stem = Path(sample).stem
            write_single_band(output / region / "pred" / f"{stem}_pred.tif", out, profile)
            save_figure(output / region / "figures" / f"{stem}_internal_frequency_compare.png", image, gt, prob >= 0.60, refined, interior, lines, carved)

    summary = {"baseline": baseline, "best": best}
    (output / f"{args.split_part}_internal_frequency_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

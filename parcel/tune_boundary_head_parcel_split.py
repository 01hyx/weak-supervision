"""Tune guarded parcel splitting driven by the learned internal-boundary head."""

import argparse
import csv
import json
from itertools import product
from pathlib import Path

import cv2
import numpy as np
import rasterio
from tqdm import tqdm

from evaluate_parcel_metrics import evaluate_case
from make_frequency_best_result_figures import read_prob
from test_manual_tiles import IGNORE_LABEL, confusion_matrix, load_sample, metrics_from_cm
from tif_binary_postprocess import write_single_band


def remove_short_lines(binary, minimum_length):
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), 8)
    output = np.zeros_like(binary)
    for label in range(1, count):
        x, y, width, height, area = stats[label]
        if area >= minimum_length and max(width, height) >= minimum_length:
            output[labels == label] = True
    return output


def refine(initial, boundary_probability, params, minimum_piece_area):
    count, labels, stats, _ = cv2.connectedComponentsWithStats(initial.astype(np.uint8), 8)
    accepted = np.zeros_like(initial)
    split_objects = 0
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] < params["minimum_object_area"]:
            continue
        obj = labels == label
        interior = cv2.erode(obj.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        candidate = interior & (boundary_probability >= params["boundary_threshold"])
        candidate = cv2.morphologyEx(
            candidate.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)
        ) > 0
        candidate = remove_short_lines(candidate, params["minimum_line_length"])
        if params["line_width"] > 1:
            candidate = cv2.dilate(
                candidate.astype(np.uint8),
                np.ones((params["line_width"], params["line_width"]), np.uint8),
            ) > 0
            candidate &= interior
        if not np.any(candidate):
            continue
        trial = obj & ~candidate
        piece_count, _, piece_stats, _ = cv2.connectedComponentsWithStats(trial.astype(np.uint8), 8)
        pieces = piece_stats[1:, cv2.CC_STAT_AREA] if piece_count > 1 else np.asarray([])
        reasonable = pieces[pieces >= minimum_piece_area]
        retained = reasonable.sum() / max(int(obj.sum()), 1)
        if len(reasonable) >= 2 and retained >= 0.92:
            accepted |= candidate
            split_objects += 1
    output = initial.copy()
    output[accepted] = False
    return output, accepted, split_objects


def aggregate(cases, params, threshold, min_area, pixel_size):
    totals = {key: 0 for key in ("gt_parcels", "pred_parcels", "matched_parcels", "over_segmented", "under_segmented")}
    cm = np.zeros((2, 2), dtype=np.int64)
    errors, offsets, predictions = [], [], []
    carved_pixels = split_objects = 0
    for case in cases:
        initial = (case["probability"] >= threshold) & case["valid"]
        if params is None:
            prediction, carved, splits = initial, np.zeros_like(initial), 0
        else:
            prediction, carved, splits = refine(
                initial, case["boundary_probability"], params, min_area
            )
        result = evaluate_case(case["gt"], prediction, min_area, 0.50, 0.10, pixel_size)
        for key in totals:
            totals[key] += result[key]
        errors.extend(result["area_errors"])
        offsets.extend(result["centroid_offsets_m"])
        out = np.full(case["label"].shape, IGNORE_LABEL, dtype=np.uint8)
        out[case["valid"]] = prediction[case["valid"]].astype(np.uint8)
        cm += confusion_matrix(out, case["label"])
        carved_pixels += int(carved.sum())
        split_objects += splits
        predictions.append(prediction)
    matched = totals["matched_parcels"]
    precision = matched / max(totals["pred_parcels"], 1)
    recall = matched / max(totals["gt_parcels"], 1)
    pixel = metrics_from_cm(cm)
    return {
        **totals,
        "parcel_precision": precision,
        "parcel_recall": recall,
        "parcel_f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "over_segmentation_rate": totals["over_segmented"] / max(totals["gt_parcels"], 1),
        "under_segmentation_rate": totals["under_segmented"] / max(totals["pred_parcels"], 1),
        "mean_area_error_percent": float(np.mean(errors) * 100) if errors else None,
        "mean_centroid_offset_m": float(np.mean(offsets)) if offsets else None,
        "target_iou": pixel["iou"][1],
        "miou": pixel["miou"],
        "carved_pixels": carved_pixels,
        "split_objects": split_objects,
    }, predictions


def load_cases(args):
    split = json.loads(Path(args.split_file).read_text(encoding="utf-8"))
    items = [item for item in split[args.split_part] if item["region"] in args.regions]
    cases = []
    for item in tqdm(items, desc="Loading boundary-head cases"):
        region, sample = item["region"], item["sample"]
        stem = Path(sample).stem
        image, label = load_sample(Path(args.data_root) / region, sample, ignore_nodata=True)
        cases.append(
            {
                "region": region,
                "sample": sample,
                "label": label,
                "valid": label != IGNORE_LABEL,
                "gt": label == 1,
                "probability": read_prob(Path(args.prob_root) / region / "prob" / f"{stem}_prob.tif"),
                "boundary_probability": read_prob(
                    Path(args.boundary_prob_root) / region / "prob" / f"{stem}_boundary_prob.tif"
                ),
            }
        )
    return cases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--prob-root", required=True)
    parser.add_argument("--boundary-prob-root", required=True)
    parser.add_argument("--split-file", required=True)
    parser.add_argument("--split-part", choices=["train", "val", "test"], required=True)
    parser.add_argument("--regions", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--threshold", type=float, default=0.60)
    parser.add_argument("--min-parcel-area", type=int, default=20)
    parser.add_argument("--pixel-size", type=float, default=10.0)
    parser.add_argument("--params-json", default=None)
    parser.add_argument("--save-best", action="store_true")
    args = parser.parse_args()

    cases = load_cases(args)
    baseline, _ = aggregate(cases, None, args.threshold, args.min_parcel_area, args.pixel_size)
    if args.params_json:
        candidates = [json.loads(Path(args.params_json).read_text(encoding="utf-8"))["best_params"]]
    else:
        candidates = [
            {
                "boundary_threshold": boundary_threshold,
                "minimum_object_area": object_area,
                "minimum_line_length": line_length,
                "line_width": line_width,
            }
            for boundary_threshold, object_area, line_length, line_width in product(
                [0.45, 0.55, 0.65, 0.75, 0.85], [150, 300, 600], [3, 6, 12], [1, 2]
            )
        ]
    rows = []
    for params in tqdm(candidates, desc="Boundary-head parcel search"):
        metrics, _ = aggregate(cases, params, args.threshold, args.min_parcel_area, args.pixel_size)
        pixel_drop = baseline["target_iou"] - metrics["target_iou"]
        over_increase = max(metrics["over_segmentation_rate"] - baseline["over_segmentation_rate"], 0)
        score = (
            metrics["parcel_f1"]
            - 0.35 * metrics["under_segmentation_rate"]
            - 0.60 * over_increase
            - 2.0 * max(pixel_drop - 0.015, 0)
        )
        rows.append({**params, **metrics, "selection_score": score})
    rows.sort(key=lambda item: item["selection_score"], reverse=True)
    best_params = {key: rows[0][key] for key in candidates[0]}
    best, predictions = aggregate(cases, best_params, args.threshold, args.min_parcel_area, args.pixel_size)

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "boundary_head_split_sweep.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    if args.save_best:
        for case, prediction in zip(cases, predictions):
            out = np.full(case["label"].shape, IGNORE_LABEL, dtype=np.uint8)
            out[case["valid"]] = prediction[case["valid"]].astype(np.uint8)
            with rasterio.open(Path(args.data_root) / case["region"] / "mask" / case["sample"]) as src:
                profile = src.profile.copy()
            stem = Path(case["sample"]).stem
            write_single_band(output / case["region"] / "pred" / f"{stem}_pred.tif", out, profile)
    summary = {
        "samples": len(cases),
        "split_part": args.split_part,
        "regions": args.regions,
        "baseline": baseline,
        "best_params": best_params,
        "best": best,
    }
    (output / "boundary_head_split_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

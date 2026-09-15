"""Reduce merged parcels with guarded marker-controlled watershed splitting."""

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
from test_manual_tiles import IGNORE_LABEL, confusion_matrix, metrics_from_cm
from tif_binary_postprocess import write_single_band
from tune_parcel_split_postprocess import load_cases


def remove_small_seed_components(seed_mask, minimum_area):
    count, labels, stats, _ = cv2.connectedComponentsWithStats(seed_mask.astype(np.uint8), 8)
    output = np.zeros_like(labels, dtype=np.int32)
    next_label = 1
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] >= minimum_area:
            output[labels == label] = next_label
            next_label += 1
    return output, next_label - 1


def internal_watershed_boundary(markers, obj):
    boundary = (markers == -1) & obj
    minimum = np.full(markers.shape, np.iinfo(np.int32).max, dtype=np.int32)
    maximum = np.zeros(markers.shape, dtype=np.int32)
    padded = np.pad(markers, 1, mode="constant", constant_values=1)
    for dy in range(3):
        for dx in range(3):
            neighbour = padded[dy : dy + markers.shape[0], dx : dx + markers.shape[1]]
            valid = neighbour > 1
            minimum[valid] = np.minimum(minimum[valid], neighbour[valid])
            maximum[valid] = np.maximum(maximum[valid], neighbour[valid])
    return boundary & (minimum < maximum)


def split_object(obj, frequency_score, seed_distance, minimum_seed_area, edge_weight, min_piece_area, max_seeds):
    distance = cv2.distanceTransform(obj.astype(np.uint8), cv2.DIST_L2, 5)
    seed_labels, seed_count = remove_small_seed_components(distance >= seed_distance, minimum_seed_area)
    if seed_count < 2 or seed_count > max_seeds:
        return np.zeros_like(obj), 0

    markers = np.zeros(obj.shape, dtype=np.int32)
    markers[~obj] = 1
    markers[seed_labels > 0] = seed_labels[seed_labels > 0] + 1
    normalized_distance = distance / max(float(distance.max()), 1e-6)
    elevation = np.clip(1.0 - normalized_distance + edge_weight * frequency_score, 0, 2)
    elevation = (elevation / max(float(elevation.max()), 1e-6) * 255).astype(np.uint8)
    cv2.watershed(np.repeat(elevation[:, :, None], 3, axis=2), markers)
    carve = internal_watershed_boundary(markers, obj)
    trial = obj & ~carve
    count, _, stats, _ = cv2.connectedComponentsWithStats(trial.astype(np.uint8), 8)
    pieces = stats[1:, cv2.CC_STAT_AREA] if count > 1 else np.asarray([])
    reasonable = pieces[pieces >= min_piece_area]
    retained = reasonable.sum() / max(int(obj.sum()), 1)
    if len(reasonable) < 2 or retained < 0.95:
        return np.zeros_like(obj), 0
    return carve, 1


def watershed_refine(initial, scores, params, min_piece_area):
    count, labels, stats, _ = cv2.connectedComponentsWithStats(initial.astype(np.uint8), 8)
    frequency = np.median(scores, axis=0).astype(np.float32)
    accepted = np.zeros_like(initial)
    split_objects = 0
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] < params["minimum_object_area"]:
            continue
        obj = labels == label
        carve, did_split = split_object(
            obj,
            frequency,
            params["seed_distance"],
            params["minimum_seed_area"],
            params["edge_weight"],
            min_piece_area,
            params["maximum_seeds"],
        )
        accepted |= carve
        split_objects += did_split
    refined = initial.copy()
    refined[accepted] = False
    return refined, accepted, split_objects


def aggregate(cases, params, threshold, min_area, pixel_size):
    totals = {key: 0 for key in ("gt_parcels", "pred_parcels", "matched_parcels", "over_segmented", "under_segmented")}
    cm = np.zeros((2, 2), dtype=np.int64)
    errors, offsets, predictions = [], [], []
    carved_pixels = split_objects = 0
    for case in cases:
        initial = (case["probability"] >= threshold) & case["valid"]
        if params is None:
            refined, carved, splits = initial, np.zeros_like(initial), 0
        else:
            refined, carved, splits = watershed_refine(initial, case["scores"], params, min_area)
        result = evaluate_case(case["gt"], refined, min_area, 0.50, 0.10, pixel_size)
        for key in totals:
            totals[key] += result[key]
        errors.extend(result["area_errors"])
        offsets.extend(result["centroid_offsets_m"])
        out = np.full(case["label"].shape, IGNORE_LABEL, dtype=np.uint8)
        out[case["valid"]] = refined[case["valid"]].astype(np.uint8)
        cm += confusion_matrix(out, case["label"])
        carved_pixels += int(carved.sum())
        split_objects += splits
        predictions.append(refined)
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--prob-root", required=True)
    parser.add_argument("--split-file", required=True)
    parser.add_argument("--split-part", choices=["train", "val", "test"], required=True)
    parser.add_argument("--regions", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--threshold", type=float, default=0.60)
    parser.add_argument("--min-parcel-area", type=int, default=20)
    parser.add_argument("--pixel-size", type=float, default=10.0)
    parser.add_argument("--params-json", default=None)
    parser.add_argument("--save-best", action="store_true")
    parser.add_argument("--aggressive-search", action="store_true")
    args = parser.parse_args()

    cases = load_cases(args)
    baseline, _ = aggregate(cases, None, args.threshold, args.min_parcel_area, args.pixel_size)
    if args.params_json:
        payload = json.loads(Path(args.params_json).read_text(encoding="utf-8"))
        candidates = [payload["best_params"]]
    else:
        if args.aggressive_search:
            search_values = ([1.0, 1.5, 2.0], [3, 5, 10], [150, 300], [0.0], [12, 20])
        else:
            search_values = ([2.0, 3.0, 4.0, 5.0], [10, 20, 40], [300, 600], [0.0, 0.5], [12])
        candidates = [
            {
                "seed_distance": distance,
                "minimum_seed_area": seed_area,
                "minimum_object_area": object_area,
                "edge_weight": edge_weight,
                "maximum_seeds": maximum_seeds,
            }
            for distance, seed_area, object_area, edge_weight, maximum_seeds in product(
                *search_values
            )
        ]

    rows = []
    for params in tqdm(candidates, desc="Watershed parcel search"):
        metrics, _ = aggregate(cases, params, args.threshold, args.min_parcel_area, args.pixel_size)
        pixel_drop = baseline["target_iou"] - metrics["target_iou"]
        over_increase = max(metrics["over_segmentation_rate"] - baseline["over_segmentation_rate"], 0)
        score = (
            metrics["parcel_f1"]
            - 0.25 * metrics["under_segmentation_rate"]
            - 0.60 * over_increase
            - 2.0 * max(pixel_drop - 0.015, 0)
        )
        rows.append({**params, **metrics, "selection_score": score})
    rows.sort(key=lambda item: item["selection_score"], reverse=True)
    best_params = {key: rows[0][key] for key in candidates[0]}
    best, predictions = aggregate(cases, best_params, args.threshold, args.min_parcel_area, args.pixel_size)

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "watershed_sweep.csv").open("w", encoding="utf-8-sig", newline="") as file:
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
    (output / "watershed_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

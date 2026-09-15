"""Evaluate parcel-level segmentation metrics from probability GeoTIFFs."""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from make_frequency_best_result_figures import read_prob
from test_manual_tiles import IGNORE_LABEL, load_sample


def connected_parcels(mask, min_area, exclude_border=True):
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    output = np.zeros_like(labels, dtype=np.int32)
    areas = []
    centers = []
    height, width = mask.shape
    next_label = 1
    for label in range(1, count):
        x, y, w, h, area = stats[label]
        touches_border = x == 0 or y == 0 or x + w >= width or y + h >= height
        if area < min_area or (exclude_border and touches_border):
            continue
        output[labels == label] = next_label
        areas.append(float(area))
        centers.append(centroids[label].astype(np.float64))
        next_label += 1
    return output, np.asarray(areas), np.asarray(centers)


def overlap_matrices(gt_labels, pred_labels, gt_areas, pred_areas):
    n_gt, n_pred = len(gt_areas), len(pred_areas)
    if n_gt == 0 or n_pred == 0:
        empty = np.zeros((n_gt, n_pred), dtype=np.float64)
        return empty, empty
    pair_ids = gt_labels.astype(np.int64) * (n_pred + 1) + pred_labels
    counts = np.bincount(pair_ids.ravel(), minlength=(n_gt + 1) * (n_pred + 1))
    intersections = counts.reshape(n_gt + 1, n_pred + 1)[1:, 1:].astype(np.float64)
    unions = gt_areas[:, None] + pred_areas[None, :] - intersections
    iou = np.divide(intersections, unions, out=np.zeros_like(intersections), where=unions > 0)
    return intersections, iou


def evaluate_case(gt, pred, min_area, match_iou, split_overlap, pixel_size):
    gt_labels, gt_areas, gt_centers = connected_parcels(gt, min_area)
    pred_labels, pred_areas, pred_centers = connected_parcels(pred, min_area)
    intersections, iou = overlap_matrices(gt_labels, pred_labels, gt_areas, pred_areas)

    matched = []
    if iou.size:
        gt_indices, pred_indices = linear_sum_assignment(1.0 - iou)
        matched = [
            (g, p)
            for g, p in zip(gt_indices, pred_indices)
            if iou[g, p] >= match_iou
        ]

    over_segmented = 0
    if len(pred_areas):
        gt_fraction = np.divide(
            intersections,
            gt_areas[:, None],
            out=np.zeros_like(intersections),
            where=gt_areas[:, None] > 0,
        )
        over_segmented = int(np.count_nonzero(np.sum(gt_fraction >= split_overlap, axis=1) >= 2))

    under_segmented = 0
    if len(gt_areas):
        pred_fraction = np.divide(
            intersections,
            pred_areas[None, :],
            out=np.zeros_like(intersections),
            where=pred_areas[None, :] > 0,
        )
        under_segmented = int(np.count_nonzero(np.sum(pred_fraction >= split_overlap, axis=0) >= 2))

    area_errors = []
    centroid_offsets = []
    matched_ious = []
    for gt_index, pred_index in matched:
        area_errors.append(abs(pred_areas[pred_index] - gt_areas[gt_index]) / gt_areas[gt_index])
        centroid_offsets.append(
            np.linalg.norm(pred_centers[pred_index] - gt_centers[gt_index]) * pixel_size
        )
        matched_ious.append(iou[gt_index, pred_index])

    return {
        "gt_parcels": len(gt_areas),
        "pred_parcels": len(pred_areas),
        "matched_parcels": len(matched),
        "over_segmented": over_segmented,
        "under_segmented": under_segmented,
        "area_errors": area_errors,
        "centroid_offsets_m": centroid_offsets,
        "matched_ious": matched_ious,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--prob-root", required=True)
    parser.add_argument("--split-file", required=True)
    parser.add_argument("--split-part", choices=["train", "val", "test"], required=True)
    parser.add_argument("--regions", nargs="+", required=True)
    parser.add_argument("--threshold", type=float, default=0.60)
    parser.add_argument("--match-iou", type=float, default=0.50)
    parser.add_argument("--split-overlap", type=float, default=0.10)
    parser.add_argument("--min-area", type=int, default=20)
    parser.add_argument("--pixel-size", type=float, default=10.0)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    split = json.loads(Path(args.split_file).read_text(encoding="utf-8"))
    selected = [item for item in split[args.split_part] if item["region"] in args.regions]
    totals = {
        "gt_parcels": 0,
        "pred_parcels": 0,
        "matched_parcels": 0,
        "over_segmented": 0,
        "under_segmented": 0,
    }
    area_errors = []
    centroid_offsets = []
    matched_ious = []
    rows = []
    for item in tqdm(selected, desc="Parcel metrics"):
        region, sample = item["region"], item["sample"]
        stem = Path(sample).stem
        _, label = load_sample(Path(args.data_root) / region, sample, ignore_nodata=True)
        valid = label != IGNORE_LABEL
        gt = (label == 1) & valid
        probability = read_prob(Path(args.prob_root) / region / "prob" / f"{stem}_prob.tif")
        pred = (probability >= args.threshold) & valid
        result = evaluate_case(
            gt,
            pred,
            args.min_area,
            args.match_iou,
            args.split_overlap,
            args.pixel_size,
        )
        for key in totals:
            totals[key] += result[key]
        area_errors.extend(result["area_errors"])
        centroid_offsets.extend(result["centroid_offsets_m"])
        matched_ious.extend(result["matched_ious"])
        rows.append({"region": region, "sample": sample, **{key: result[key] for key in totals}})

    matched = totals["matched_parcels"]
    precision = matched / max(totals["pred_parcels"], 1)
    recall = matched / max(totals["gt_parcels"], 1)
    summary = {
        "samples": len(selected),
        "split_part": args.split_part,
        "regions": args.regions,
        "threshold": args.threshold,
        "match_iou": args.match_iou,
        "minimum_parcel_pixels": args.min_area,
        "minimum_parcel_hectares": args.min_area * args.pixel_size**2 / 10000.0,
        "edge_touching_parcels_excluded": True,
        **totals,
        "parcel_precision": precision,
        "parcel_recall": recall,
        "parcel_f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "over_segmentation_rate": totals["over_segmented"] / max(totals["gt_parcels"], 1),
        "under_segmentation_rate": totals["under_segmented"] / max(totals["pred_parcels"], 1),
        "mean_absolute_area_error_percent": float(np.mean(area_errors) * 100) if area_errors else None,
        "mean_centroid_offset_m": float(np.mean(centroid_offsets)) if centroid_offsets else None,
        "mean_matched_iou": float(np.mean(matched_ious)) if matched_ious else None,
    }

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "parcel_metrics_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (output / "parcel_metrics_per_sample.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

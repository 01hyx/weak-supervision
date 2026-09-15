"""Decode center/offset maps into parcel instances and evaluate object metrics."""

import argparse
import csv
import json
from itertools import product
from pathlib import Path

import cv2
import numpy as np
import rasterio
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from tqdm import tqdm

from tif_binary_postprocess import write_single_band


def read_float(path):
    with rasterio.open(path) as src:
        return src.read(1).astype(np.float32)


def filtered_relabel(labels, minimum_area=20, exclude_border=True):
    output = np.zeros(labels.shape, dtype=np.int32)
    areas, centers = [], []
    next_label = 1
    height, width = labels.shape
    for value in np.unique(labels):
        if value == 0:
            continue
        mask = labels == value
        ys, xs = np.nonzero(mask)
        if len(xs) < minimum_area:
            continue
        if exclude_border and (
            np.any(xs == 0) or np.any(ys == 0) or np.any(xs == width - 1) or np.any(ys == height - 1)
        ):
            continue
        output[mask] = next_label
        areas.append(float(len(xs)))
        centers.append((float(xs.mean()), float(ys.mean())))
        next_label += 1
    return output, np.asarray(areas), np.asarray(centers)


def instance_metrics(gt_raw, pred_raw, minimum_area=20):
    gt, gt_areas, gt_centers = filtered_relabel(gt_raw, minimum_area)
    pred, pred_areas, pred_centers = filtered_relabel(pred_raw, minimum_area)
    n_gt, n_pred = len(gt_areas), len(pred_areas)
    intersections = np.zeros((n_gt, n_pred), dtype=np.float64)
    if n_gt and n_pred:
        pair = gt.astype(np.int64) * (n_pred + 1) + pred
        counts = np.bincount(pair.ravel(), minlength=(n_gt + 1) * (n_pred + 1))
        intersections = counts.reshape(n_gt + 1, n_pred + 1)[1:, 1:].astype(np.float64)
    unions = gt_areas[:, None] + pred_areas[None, :] - intersections
    iou = np.divide(intersections, unions, out=np.zeros_like(intersections), where=unions > 0)
    matches = []
    if iou.size:
        rows, cols = linear_sum_assignment(1.0 - iou)
        matches = [(g, p) for g, p in zip(rows, cols) if iou[g, p] >= 0.50]
    over = int(
        np.count_nonzero(
            np.sum(
                np.divide(intersections, gt_areas[:, None], out=np.zeros_like(intersections), where=gt_areas[:, None] > 0)
                >= 0.10,
                axis=1,
            )
            >= 2
        )
    ) if n_gt else 0
    under = int(
        np.count_nonzero(
            np.sum(
                np.divide(intersections, pred_areas[None, :], out=np.zeros_like(intersections), where=pred_areas[None, :] > 0)
                >= 0.10,
                axis=0,
            )
            >= 2
        )
    ) if n_pred else 0
    area_errors, offsets = [], []
    for g, p in matches:
        area_errors.append(abs(pred_areas[p] - gt_areas[g]) / gt_areas[g])
        offsets.append(np.linalg.norm(pred_centers[p] - gt_centers[g]) * 10.0)
    return {
        "gt_parcels": n_gt,
        "pred_parcels": n_pred,
        "matched_parcels": len(matches),
        "over_segmented": over,
        "under_segmented": under,
        "area_errors": area_errors,
        "centroid_offsets_m": offsets,
    }


def detect_centers(center, semantic_mask, threshold, nms_radius):
    kernel = np.ones((nms_radius * 2 + 1, nms_radius * 2 + 1), np.uint8)
    maximum = cv2.dilate(center.astype(np.float32), kernel)
    peaks = semantic_mask & (center >= threshold) & (center >= maximum - 1e-6)
    count, labels, _, _ = cv2.connectedComponentsWithStats(peaks.astype(np.uint8), 8)
    centers = []
    for label in range(1, count):
        ys, xs = np.nonzero(labels == label)
        if len(xs):
            best = np.argmax(center[ys, xs])
            centers.append((float(xs[best]), float(ys[best]), float(center[ys[best], xs[best]])))
    return centers


def decode_instances(semantic, center, offset_x, offset_y, params):
    mask = semantic >= params["semantic_threshold"]
    center_candidates = detect_centers(
        center, mask, params["center_threshold"], params["nms_radius"]
    )
    count, components, _, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    instances = np.zeros(mask.shape, dtype=np.uint32)
    next_id = 1
    height, width = mask.shape
    for component in range(1, count):
        obj = components == component
        ys, xs = np.nonzero(obj)
        local_centers = [item for item in center_candidates if obj[int(round(item[1])), int(round(item[0]))]]
        if not local_centers:
            instances[obj] = next_id
            next_id += 1
            continue
        local_centers.sort(key=lambda item: item[2], reverse=True)
        local_centers = local_centers[: params["maximum_centers"]]
        coordinates = np.column_stack([xs, ys]).astype(np.float32)
        shifted = coordinates.copy()
        shifted[:, 0] += params["offset_weight"] * offset_x[ys, xs] * width
        shifted[:, 1] += params["offset_weight"] * offset_y[ys, xs] * height
        tree = cKDTree(np.asarray([[item[0], item[1]] for item in local_centers]))
        _, assignment = tree.query(shifted, k=1)
        for local_id in range(len(local_centers)):
            selected = assignment == local_id
            if np.any(selected):
                instances[ys[selected], xs[selected]] = next_id
                next_id += 1
    return instances


def aggregate(cases, params, minimum_area=20):
    totals = {key: 0 for key in ("gt_parcels", "pred_parcels", "matched_parcels", "over_segmented", "under_segmented")}
    errors, offsets, predictions = [], [], []
    for case in cases:
        pred = decode_instances(
            case["semantic"], case["center"], case["offset_x"], case["offset_y"], params
        )
        metrics = instance_metrics(case["ground_truth"], pred, minimum_area)
        for key in totals:
            totals[key] += metrics[key]
        errors.extend(metrics["area_errors"])
        offsets.extend(metrics["centroid_offsets_m"])
        predictions.append(pred)
    matched = totals["matched_parcels"]
    precision = matched / max(totals["pred_parcels"], 1)
    recall = matched / max(totals["gt_parcels"], 1)
    return {
        **totals,
        "parcel_precision": precision,
        "parcel_recall": recall,
        "parcel_f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "over_segmentation_rate": totals["over_segmented"] / max(totals["gt_parcels"], 1),
        "under_segmentation_rate": totals["under_segmented"] / max(totals["pred_parcels"], 1),
        "mean_area_error_percent": float(np.mean(errors) * 100) if errors else None,
        "mean_centroid_offset_m": float(np.mean(offsets)) if offsets else None,
    }, predictions


def load_cases(args):
    split = json.loads(Path(args.split_file).read_text(encoding="utf-8"))
    items = [item for item in split[args.split_part] if item["region"] == args.region]
    cases = []
    region_output = Path(args.output_root) / args.region
    for item in tqdm(items, desc="Loading instance outputs"):
        sample = item["sample"]
        stem = Path(sample).stem
        with rasterio.open(Path(args.data_root) / args.region / "instance_mask" / sample) as src:
            ground_truth = src.read(1)
        cases.append(
            {
                "sample": sample,
                "ground_truth": ground_truth,
                "semantic": read_float(region_output / "semantic" / f"{stem}_semantic.tif"),
                "center": read_float(region_output / "center" / f"{stem}_center.tif"),
                "offset_x": read_float(region_output / "offset_x" / f"{stem}_offset_x.tif"),
                "offset_y": read_float(region_output / "offset_y" / f"{stem}_offset_y.tif"),
            }
        )
    return cases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=r"D:\s2_output\qingdao_selected150_2025_sentinel")
    parser.add_argument("--output-root", default="qingdao_parcel_instance_outputs")
    parser.add_argument("--split-file", default="configs/qingdao_2025_spatial_split.json")
    parser.add_argument("--split-part", choices=["val", "test"], required=True)
    parser.add_argument("--region", default="青岛夏玉米0925")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--params-json", default=None)
    parser.add_argument("--minimum-area", type=int, default=20)
    parser.add_argument("--save-best", action="store_true")
    args = parser.parse_args()

    cases = load_cases(args)
    if args.params_json:
        candidates = [json.loads(Path(args.params_json).read_text(encoding="utf-8"))["best_params"]]
    else:
        candidates = [
            {
                "semantic_threshold": 0.60,
                "center_threshold": center_threshold,
                "nms_radius": nms_radius,
                "offset_weight": offset_weight,
                "maximum_centers": 80,
            }
            for center_threshold, nms_radius, offset_weight in product(
                [0.45, 0.50, 0.55, 0.60, 0.65, 0.70], [2, 4, 6], [0.0, 0.5, 1.0]
            )
        ]
    rows = []
    for params in tqdm(candidates, desc="Instance decode search"):
        metrics, _ = aggregate(cases, params, args.minimum_area)
        score = (
            metrics["parcel_f1"]
            - 0.25 * metrics["under_segmentation_rate"]
            - 0.40 * metrics["over_segmentation_rate"]
        )
        rows.append({**params, **metrics, "selection_score": score})
    rows.sort(key=lambda item: item["selection_score"], reverse=True)
    best_params = {key: rows[0][key] for key in candidates[0]}
    best, predictions = aggregate(cases, best_params, args.minimum_area)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "instance_decode_sweep.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    if args.save_best:
        for case, prediction in zip(cases, predictions):
            sample = case["sample"]
            with rasterio.open(Path(args.data_root) / args.region / "instance_mask" / sample) as src:
                profile = src.profile.copy()
            stem = Path(sample).stem
            write_single_band(
                output / args.region / "instance" / f"{stem}_instance.tif",
                prediction,
                profile,
                dtype="uint32",
                nodata=0,
            )
    summary = {
        "samples": len(cases),
        "split_part": args.split_part,
        "minimum_area": args.minimum_area,
        "best_params": best_params,
        "best": best,
    }
    (output / "instance_metrics_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

"""统一评价基础模型、区域微调与频域模型的边界精度。"""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from make_frequency_best_result_figures import read_prob
from test_manual_tiles import IGNORE_LABEL, load_sample


MODELS = {
    "base": {
        "root": "manual_tile_prediction_tifs_all",
        "threshold": 0.12,
    },
    "adapted": {
        "root": "manual_tile_prediction_tifs_finetune_head",
        "threshold": 0.65,
    },
    "frequency": {
        "root": "frequency_prediction_tifs_all",
        "threshold": 0.60,
    },
    "frequency_boundary": {
        "root": "frequency_boundary_prediction_tifs_all",
        "threshold": 0.60,
    },
}


def binary_boundary(mask, valid, radius=1):
    """提取二值区域的内外形态学边界，并排除无效数据附近的伪边界。"""
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1)
    )
    mask_u8 = (mask & valid).astype(np.uint8)
    dilated = cv2.dilate(mask_u8, kernel) > 0
    eroded = cv2.erode(mask_u8, kernel) > 0
    boundary = dilated != eroded

    valid_u8 = valid.astype(np.uint8)
    valid_core = cv2.erode(valid_u8, kernel) > 0
    return boundary & valid_core


def dilate(mask, tolerance):
    if tolerance <= 0:
        return mask
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (tolerance * 2 + 1, tolerance * 2 + 1)
    )
    return cv2.dilate(mask.astype(np.uint8), kernel) > 0


def boundary_match_counts(gt_boundary, pred_boundary, tolerance):
    gt_band = dilate(gt_boundary, tolerance)
    pred_band = dilate(pred_boundary, tolerance)
    matched_pred = int(np.count_nonzero(pred_boundary & gt_band))
    matched_gt = int(np.count_nonzero(gt_boundary & pred_band))
    return {
        "matched_pred": matched_pred,
        "pred_boundary": int(np.count_nonzero(pred_boundary)),
        "matched_gt": matched_gt,
        "gt_boundary": int(np.count_nonzero(gt_boundary)),
    }


def boundary_iou(gt_boundary, pred_boundary, tolerance):
    gt_band = dilate(gt_boundary, tolerance)
    pred_band = dilate(pred_boundary, tolerance)
    intersection = int(np.count_nonzero(gt_band & pred_band))
    union = int(np.count_nonzero(gt_band | pred_band))
    return intersection, union


def boundary_distances(gt_boundary, pred_boundary):
    """计算双向边界距离，单位为像素。"""
    if not np.any(gt_boundary) or not np.any(pred_boundary):
        return np.array([], dtype=np.float32)
    distance_to_gt = cv2.distanceTransform(
        (~gt_boundary).astype(np.uint8), cv2.DIST_L2, 5
    )
    distance_to_pred = cv2.distanceTransform(
        (~pred_boundary).astype(np.uint8), cv2.DIST_L2, 5
    )
    return np.concatenate(
        [distance_to_gt[pred_boundary], distance_to_pred[gt_boundary]]
    ).astype(np.float32)


def empty_accumulator(tolerances):
    return {
        "counts": {
            str(t): {
                "matched_pred": 0,
                "pred_boundary": 0,
                "matched_gt": 0,
                "gt_boundary": 0,
                "iou_intersection": 0,
                "iou_union": 0,
            }
            for t in tolerances
        },
        "distance_sum": 0.0,
        "distance_count": 0,
        "sample_assd": [],
        "sample_hd95": [],
        "samples": 0,
    }


def metrics_from_accumulator(accumulator, tolerance, pixel_size):
    counts = accumulator["counts"][str(tolerance)]
    precision = counts["matched_pred"] / max(counts["pred_boundary"], 1)
    recall = counts["matched_gt"] / max(counts["gt_boundary"], 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    biou = counts["iou_intersection"] / max(counts["iou_union"], 1)
    mean_distance = accumulator["distance_sum"] / max(accumulator["distance_count"], 1)
    return {
        "boundary_precision": precision,
        "boundary_recall": recall,
        "boundary_f1": f1,
        "boundary_iou": biou,
        "mean_boundary_distance_px": mean_distance,
        "mean_boundary_distance_m": mean_distance * pixel_size,
        "mean_sample_assd_px": float(np.mean(accumulator["sample_assd"])),
        "mean_sample_assd_m": float(np.mean(accumulator["sample_assd"])) * pixel_size,
        "mean_sample_hd95_px": float(np.mean(accumulator["sample_hd95"])),
        "mean_sample_hd95_m": float(np.mean(accumulator["sample_hd95"])) * pixel_size,
    }


def evaluate_sample(gt, pred, valid, tolerances, boundary_radius):
    gt_boundary = binary_boundary(gt, valid, boundary_radius)
    pred_boundary = binary_boundary(pred, valid, boundary_radius)
    result = {"counts": {}}
    for tolerance in tolerances:
        counts = boundary_match_counts(gt_boundary, pred_boundary, tolerance)
        intersection, union = boundary_iou(gt_boundary, pred_boundary, tolerance)
        counts["iou_intersection"] = intersection
        counts["iou_union"] = union
        result["counts"][str(tolerance)] = counts
    distances = boundary_distances(gt_boundary, pred_boundary)
    result["assd_px"] = float(np.mean(distances)) if distances.size else np.nan
    result["hd95_px"] = float(np.percentile(distances, 95)) if distances.size else np.nan
    result["gt_boundary_pixels"] = int(gt_boundary.sum())
    result["pred_boundary_pixels"] = int(pred_boundary.sum())
    return result


def add_sample(accumulator, sample_result):
    for tolerance, counts in sample_result["counts"].items():
        for key, value in counts.items():
            accumulator["counts"][tolerance][key] += value
    if np.isfinite(sample_result["assd_px"]):
        total_boundary = (
            sample_result["gt_boundary_pixels"] + sample_result["pred_boundary_pixels"]
        )
        accumulator["distance_sum"] += sample_result["assd_px"] * total_boundary
        accumulator["distance_count"] += total_boundary
        accumulator["sample_assd"].append(sample_result["assd_px"])
        accumulator["sample_hd95"].append(sample_result["hd95_px"])
    accumulator["samples"] += 1


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate boundary segmentation metrics.")
    parser.add_argument(
        "--split",
        default="manual_tiles_filtered_problem_removed/filtered_trainval_split.json",
    )
    parser.add_argument(
        "--data-root", default=r"D:\s2_output\manual_tiles_maize30_stride128"
    )
    parser.add_argument(
        "--output-dir", default="boundary_metrics_all_models"
    )
    parser.add_argument("--tolerances", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--boundary-radius", type=int, default=1)
    parser.add_argument("--pixel-size", type=float, default=10.0)
    return parser.parse_args()


def main():
    args = parse_args()
    split = json.loads(Path(args.split).read_text(encoding="utf-8"))
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    accumulators = {
        part: {
            model: empty_accumulator(args.tolerances)
            for model in MODELS
        }
        for part in ("train", "val", "all")
    }
    sample_rows = []

    for part in ("train", "val"):
        for item in tqdm(split[part], desc=f"boundary metrics {part}"):
            region, sample = item["region"], item["sample"]
            stem = Path(sample).stem
            _, label = load_sample(
                Path(args.data_root) / region, sample, ignore_nodata=True
            )
            valid = label != IGNORE_LABEL
            gt = label == 1

            for model, config in MODELS.items():
                prob_path = (
                    Path(config["root"]) / region / "prob" / f"{stem}_prob.tif"
                )
                pred = (read_prob(prob_path) >= config["threshold"]) & valid
                result = evaluate_sample(
                    gt, pred, valid, args.tolerances, args.boundary_radius
                )
                add_sample(accumulators[part][model], result)
                add_sample(accumulators["all"][model], result)

                row = {
                    "split": part,
                    "region": region,
                    "sample": sample,
                    "model": model,
                    "threshold": config["threshold"],
                    "assd_px": result["assd_px"],
                    "assd_m": result["assd_px"] * args.pixel_size,
                    "hd95_px": result["hd95_px"],
                    "hd95_m": result["hd95_px"] * args.pixel_size,
                    "gt_boundary_pixels": result["gt_boundary_pixels"],
                    "pred_boundary_pixels": result["pred_boundary_pixels"],
                }
                for tolerance in args.tolerances:
                    counts = result["counts"][str(tolerance)]
                    precision = counts["matched_pred"] / max(
                        counts["pred_boundary"], 1
                    )
                    recall = counts["matched_gt"] / max(
                        counts["gt_boundary"], 1
                    )
                    row[f"bp_t{tolerance}"] = precision
                    row[f"br_t{tolerance}"] = recall
                    row[f"bf1_t{tolerance}"] = (
                        2 * precision * recall / max(precision + recall, 1e-12)
                    )
                    row[f"biou_t{tolerance}"] = (
                        counts["iou_intersection"] / max(counts["iou_union"], 1)
                    )
                sample_rows.append(row)

    summary_rows = []
    summary_json = {}
    for part in ("train", "val", "all"):
        summary_json[part] = {}
        for model, accumulator in accumulators[part].items():
            summary_json[part][model] = {}
            for tolerance in args.tolerances:
                metrics = metrics_from_accumulator(
                    accumulator, tolerance, args.pixel_size
                )
                summary_json[part][model][str(tolerance)] = metrics
                summary_rows.append(
                    {
                        "split": part,
                        "model": model,
                        "samples": accumulator["samples"],
                        "tolerance_px": tolerance,
                        "tolerance_m": tolerance * args.pixel_size,
                        **metrics,
                    }
                )

    with (output / "boundary_metrics_summary.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    with (output / "boundary_metrics_per_sample.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(sample_rows[0].keys()))
        writer.writeheader()
        writer.writerows(sample_rows)
    (output / "boundary_metrics_summary.json").write_text(
        json.dumps(summary_json, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(json.dumps(summary_json["val"], ensure_ascii=False, indent=2))
    print(f"[INFO] Summary: {(output / 'boundary_metrics_summary.csv').resolve()}")
    print(f"[INFO] Per sample: {(output / 'boundary_metrics_per_sample.csv').resolve()}")


if __name__ == "__main__":
    main()

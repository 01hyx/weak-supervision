import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from test_manual_tiles import DEFAULT_DATA_ROOT, IGNORE_LABEL, confusion_matrix, metrics_from_cm
from tif_binary_postprocess import postprocess, read_single_band, write_single_band


def load_split_samples(split_file, split_name):
    with Path(split_file).open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload[split_name]


def mask_path(data_root, region, sample):
    return Path(data_root) / region / "mask" / sample


def read_case(data_root, pred_root, item):
    region = item["region"]
    sample = item["sample"]
    stem = Path(sample).stem
    pred_path = Path(pred_root) / region / "pred" / f"{stem}_pred.tif"
    prob_path = Path(pred_root) / region / "prob" / f"{stem}_prob.tif"
    gt_path = mask_path(data_root, region, sample)

    pred_arr, profile = read_single_band(pred_path)
    prob, _ = read_single_band(prob_path)
    gt, _ = read_single_band(gt_path)
    gt = (gt > 0).astype(np.int64)
    gt[pred_arr == IGNORE_LABEL] = IGNORE_LABEL
    prob = prob.astype(np.float32)
    if prob.max() > 1.0:
        prob = prob / 255.0
    return region, stem, pred_arr, prob, gt, profile


def evaluate_prediction(pred_arr, gt):
    cm = confusion_matrix(pred_arr.astype(np.uint8), gt.astype(np.int64))
    metrics = metrics_from_cm(cm)
    return cm, metrics


def make_args(min_area, max_hole_area):
    return SimpleNamespace(
        candidate_prob_threshold=1.1,
        overlap_threshold=1.1,
        prob_threshold=1.1,
        min_object_area=min_area,
        min_component_area=min_area,
        max_hole_area=max_hole_area,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Conservative postprocess sweep on a saved split.")
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--pred-root", default="manual_tile_prediction_tifs_finetune_head")
    parser.add_argument("--split-file", default="manual_tiles_finetune_head_trial3/split.json")
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--output-dir", default="manual_tile_finetune_conservative_postprocess")
    parser.add_argument("--min-areas", default="10,20,30,50,80,120,160,220")
    parser.add_argument("--hole-areas", default="0,16,32,64,128,256")
    parser.add_argument("--thresholds", default="0.60,0.625,0.65,0.675,0.70,0.725,0.75")
    parser.add_argument("--save-best-tifs", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    samples = load_split_samples(args.split_file, args.split)
    min_areas = [int(x.strip()) for x in args.min_areas.split(",") if x.strip()]
    hole_areas = [int(x.strip()) for x in args.hole_areas.split(",") if x.strip()]
    thresholds = [float(x.strip()) for x in args.thresholds.split(",") if x.strip()]
    cases = [read_case(args.data_root, args.pred_root, item) for item in samples]

    baseline_cm = np.zeros((2, 2), dtype=np.int64)
    baseline_threshold = 0.70 if 0.70 in thresholds else thresholds[0]
    for _, _, pred_arr, prob, gt, _ in cases:
        pred_arr = np.full(pred_arr.shape, IGNORE_LABEL, dtype=np.uint8)
        valid = gt != IGNORE_LABEL
        pred_arr[valid] = (prob[valid] >= baseline_threshold).astype(np.uint8)
        cm, _ = evaluate_prediction(pred_arr, gt)
        baseline_cm += cm
    baseline = metrics_from_cm(baseline_cm)

    rows = []
    best_row = None
    best_cm = None
    for threshold in thresholds:
        for min_area in min_areas:
            for hole_area in hole_areas:
                cm_total = np.zeros((2, 2), dtype=np.int64)
                pp_args = make_args(min_area, hole_area)
                for _, _, pred_arr, prob, gt, _ in cases:
                    valid = gt != IGNORE_LABEL
                    threshold_pred = prob >= threshold
                    refined = postprocess(threshold_pred, prob, None, pp_args)
                    out = np.full(pred_arr.shape, IGNORE_LABEL, dtype=np.uint8)
                    out[valid] = refined[valid]
                    cm, _ = evaluate_prediction(out, gt)
                    cm_total += cm

                metrics = metrics_from_cm(cm_total)
                row = {
                    "threshold": threshold,
                    "min_area": min_area,
                    "max_hole_area": hole_area,
                    "pixels": metrics["pixels"],
                    "oa": metrics["oa"],
                    "miou": metrics["miou"],
                    "target_iou": metrics["iou"][1],
                    "target_f1": metrics["f1"][1],
                    "fp": int(cm_total[0, 1]),
                    "fn": int(cm_total[1, 0]),
                    "confusion_matrix": cm_total.tolist(),
                }
                rows.append(row)
                if best_row is None or row["miou"] > best_row["miou"]:
                    best_row = row
                    best_cm = cm_total.copy()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{args.split}_conservative_sweep.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    if args.save_best_tifs:
        pp_args = make_args(best_row["min_area"], best_row["max_hole_area"])
        for region, stem, pred_arr, prob, gt, profile in cases:
            valid = gt != IGNORE_LABEL
            refined = postprocess(prob >= best_row["threshold"], prob, None, pp_args)
            out = np.full(pred_arr.shape, IGNORE_LABEL, dtype=np.uint8)
            out[valid] = refined[valid]
            write_single_band(output_dir / region / f"{stem}_pred_conservative_post.tif", out, profile)

    print("BASELINE", {
        "cm": baseline_cm.tolist(),
        "oa": baseline["oa"],
        "miou": baseline["miou"],
        "target_iou": baseline["iou"][1],
        "target_f1": baseline["f1"][1],
        "fp": int(baseline_cm[0, 1]),
        "fn": int(baseline_cm[1, 0]),
    })
    print("BEST", best_row)
    print(f"[INFO] Saved sweep: {csv_path}")


if __name__ == "__main__":
    main()

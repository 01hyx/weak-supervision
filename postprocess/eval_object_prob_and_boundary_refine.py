import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from test_manual_tiles import DEFAULT_DATA_ROOT, IGNORE_LABEL, confusion_matrix, metrics_from_cm, load_multitemporal_image
from tif_binary_postprocess import connected_components, read_single_band, write_single_band


def load_split_samples(split_file, split_name):
    with Path(split_file).open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload[split_name]


def read_case(data_root, pred_root, item):
    region = item["region"]
    sample = item["sample"]
    stem = Path(sample).stem
    region_dir = Path(data_root) / region
    prob_path = Path(pred_root) / region / "prob" / f"{stem}_prob.tif"
    mask_path = region_dir / "mask" / sample
    prob, profile = read_single_band(prob_path)
    gt, _ = read_single_band(mask_path)
    image, valid_mask = load_multitemporal_image(region_dir, sample)
    gt = (gt > 0).astype(np.int64)
    gt[~valid_mask] = IGNORE_LABEL
    prob = prob.astype(np.float32)
    if prob.max() > 1.0:
        prob = prob / 255.0
    return region, stem, image, prob, gt, profile


def object_probability_filter(prob, threshold, min_area, min_mean_prob, min_max_prob, min_high_ratio, high_prob):
    pred = prob >= threshold
    labels, components = connected_components(pred)
    out = pred.copy()
    for idx, coords in enumerate(components, start=1):
        obj = labels == idx
        values = prob[obj]
        area = int(values.size)
        mean_prob = float(values.mean()) if area else 0.0
        max_prob = float(values.max()) if area else 0.0
        high_ratio = float(np.mean(values >= high_prob)) if area else 0.0
        if area < min_area or mean_prob < min_mean_prob or max_prob < min_max_prob or high_ratio < min_high_ratio:
            out[obj] = False
    return out


def make_rgb_u8(image):
    # Each time step has [B2, B3, B4, B8, B11, B12]. Use the clearest late-season RGB proxy.
    data = image.reshape(6, 6, image.shape[-2], image.shape[-1])
    rgb = np.stack([data[4, 2], data[4, 1], data[4, 0]], axis=-1)
    out = np.zeros_like(rgb, dtype=np.float32)
    for channel in range(3):
        vals = rgb[..., channel]
        lo, hi = np.percentile(vals[vals > 0], [2, 98]) if np.any(vals > 0) else (0.0, 1.0)
        out[..., channel] = np.clip((vals - lo) / max(hi - lo, 1e-6), 0, 1)
    return (out * 255).astype(np.uint8)


def boundary_smooth_prob(image, prob, diameter, sigma_color, sigma_space, blend):
    rgb = make_rgb_u8(image)
    prob_u8 = np.clip(prob * 255.0, 0, 255).astype(np.uint8)
    smooth = cv2.bilateralFilter(prob_u8, diameter, sigma_color, sigma_space).astype(np.float32) / 255.0
    # Reinject a little original probability to avoid over-smoothing small real fields.
    return (1.0 - blend) * prob + blend * smooth


def eval_pred(pred, gt):
    pred_arr = np.full(gt.shape, IGNORE_LABEL, dtype=np.uint8)
    valid = gt != IGNORE_LABEL
    pred_arr[valid] = pred[valid].astype(np.uint8)
    cm = confusion_matrix(pred_arr, gt)
    return cm


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate object probability filtering and boundary smoothing.")
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--pred-root", default="manual_tile_prediction_tifs_finetune_head")
    parser.add_argument("--split-file", default="manual_tiles_finetune_head_trial3/split.json")
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--output-dir", default="manual_tile_object_prob_boundary_refine")
    parser.add_argument("--thresholds", default="0.65,0.675,0.70,0.725")
    parser.add_argument("--min-areas", default="10,20,50,80")
    parser.add_argument("--min-mean-probs", default="0.68,0.70,0.72,0.75")
    parser.add_argument("--min-max-probs", default="0.75,0.80,0.85")
    parser.add_argument("--min-high-ratios", default="0.0,0.05,0.10")
    parser.add_argument("--high-prob", type=float, default=0.85)
    parser.add_argument("--save-best-tifs", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    samples = load_split_samples(args.split_file, args.split)
    cases = [read_case(args.data_root, args.pred_root, item) for item in samples]

    thresholds = [float(x) for x in args.thresholds.split(",") if x]
    min_areas = [int(x) for x in args.min_areas.split(",") if x]
    min_means = [float(x) for x in args.min_mean_probs.split(",") if x]
    min_maxes = [float(x) for x in args.min_max_probs.split(",") if x]
    min_high_ratios = [float(x) for x in args.min_high_ratios.split(",") if x]

    rows = []
    best = None
    for threshold in thresholds:
        cm_base = np.zeros((2, 2), dtype=np.int64)
        cm_boundary = np.zeros((2, 2), dtype=np.int64)
        for _, _, image, prob, gt, _ in cases:
            cm_base += eval_pred(prob >= threshold, gt)
            smooth_prob = boundary_smooth_prob(image, prob, 5, 25, 9, 0.45)
            cm_boundary += eval_pred(smooth_prob >= threshold, gt)
        for method, cm in [("threshold", cm_base), ("boundary_smooth", cm_boundary)]:
            metrics = metrics_from_cm(cm)
            row = {
                "method": method,
                "threshold": threshold,
                "min_area": 0,
                "min_mean_prob": 0.0,
                "min_max_prob": 0.0,
                "min_high_ratio": 0.0,
                "oa": metrics["oa"],
                "miou": metrics["miou"],
                "target_iou": metrics["iou"][1],
                "target_f1": metrics["f1"][1],
                "fp": int(cm[0, 1]),
                "fn": int(cm[1, 0]),
                "confusion_matrix": cm.tolist(),
            }
            rows.append(row)
            if best is None or row["miou"] > best["miou"]:
                best = row

    for threshold in thresholds:
        for min_area in min_areas:
            for min_mean in min_means:
                for min_max in min_maxes:
                    for min_high_ratio in min_high_ratios:
                        cm_total = np.zeros((2, 2), dtype=np.int64)
                        for _, _, _, prob, gt, _ in cases:
                            pred = object_probability_filter(
                                prob, threshold, min_area, min_mean, min_max, min_high_ratio, args.high_prob
                            )
                            cm_total += eval_pred(pred, gt)
                        metrics = metrics_from_cm(cm_total)
                        row = {
                            "method": "object_prob",
                            "threshold": threshold,
                            "min_area": min_area,
                            "min_mean_prob": min_mean,
                            "min_max_prob": min_max,
                            "min_high_ratio": min_high_ratio,
                            "oa": metrics["oa"],
                            "miou": metrics["miou"],
                            "target_iou": metrics["iou"][1],
                            "target_f1": metrics["f1"][1],
                            "fp": int(cm_total[0, 1]),
                            "fn": int(cm_total[1, 0]),
                            "confusion_matrix": cm_total.tolist(),
                        }
                        rows.append(row)
                        if row["miou"] > best["miou"]:
                            best = row

    csv_path = output_dir / f"{args.split}_object_prob_boundary_sweep.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    if args.save_best_tifs and best["method"] == "object_prob":
        for region, stem, _, prob, gt, profile in cases:
            pred = object_probability_filter(
                prob,
                best["threshold"],
                int(best["min_area"]),
                best["min_mean_prob"],
                best["min_max_prob"],
                best["min_high_ratio"],
                args.high_prob,
            )
            out = np.full(gt.shape, IGNORE_LABEL, dtype=np.uint8)
            valid = gt != IGNORE_LABEL
            out[valid] = pred[valid].astype(np.uint8)
            write_single_band(output_dir / region / f"{stem}_pred_object_prob.tif", out, profile)

    print("BEST", best)
    print(f"[INFO] Saved: {csv_path}")


if __name__ == "__main__":
    main()

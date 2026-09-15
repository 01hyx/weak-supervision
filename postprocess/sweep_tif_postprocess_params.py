import argparse
import csv
from itertools import product
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from batch_eval_tif_postprocess import find_mask, row_from_metrics
from test_manual_tiles import DEFAULT_DATA_ROOT, IGNORE_LABEL, confusion_matrix, metrics_from_cm
from tif_binary_postprocess import make_candidate_object_labels, postprocess, read_single_band, sessrs_object_refine


def parse_list(text, cast=float):
    return [cast(item.strip()) for item in text.split(",") if item.strip()]


def evaluate_variant(args, params, samples):
    overall_sessrs = np.zeros((2, 2), dtype=np.int64)
    overall_post = np.zeros((2, 2), dtype=np.int64)
    config = SimpleNamespace(
        prob_threshold=params["prob_threshold"],
        candidate_prob_threshold=params["candidate_prob_threshold"],
        overlap_threshold=params["overlap_threshold"],
        min_object_area=params["min_object_area"],
        min_component_area=params["min_component_area"],
        max_hole_area=params["max_hole_area"],
    )

    for sample in samples:
        pred_arr, prob, gt, label_cache = sample
        pred = pred_arr == 1
        valid = pred_arr != args.nodata
        cache_key = params["candidate_prob_threshold"]
        if cache_key not in label_cache:
            label_cache[cache_key] = make_candidate_object_labels(pred, prob, cache_key)
        object_labels = label_cache[cache_key]

        sessrs = sessrs_object_refine(pred, prob, object_labels, config).astype(np.uint8)
        sessrs_out = np.full(pred_arr.shape, args.nodata, dtype=np.uint8)
        sessrs_out[valid] = sessrs[valid]

        post = postprocess(pred, prob, object_labels, config).astype(np.uint8)
        post_out = np.full(pred_arr.shape, args.nodata, dtype=np.uint8)
        post_out[valid] = post[valid]

        overall_sessrs += confusion_matrix(sessrs_out, gt)
        overall_post += confusion_matrix(post_out, gt)

    sessrs_metrics = metrics_from_cm(overall_sessrs)
    post_metrics = metrics_from_cm(overall_post)
    return sessrs_metrics, post_metrics


def load_samples(args):
    samples = []
    pred_root = Path(args.pred_root) / args.region
    for stem in args.stems:
        pred_path = pred_root / "pred" / f"{stem}_pred.tif"
        prob_path = pred_root / "prob" / f"{stem}_prob.tif"
        mask_path = find_mask(args.data_root, args.region, stem)

        pred_arr, _ = read_single_band(pred_path)
        prob, _ = read_single_band(prob_path)
        gt, _ = read_single_band(mask_path)
        gt = (gt > 0).astype(np.int64)
        gt[pred_arr == args.nodata] = args.nodata
        prob = prob.astype(np.float32)
        if prob.max() > 1.0:
            prob = prob / 255.0
        samples.append((pred_arr, prob, gt, {}))
    return samples


def parse_args():
    parser = argparse.ArgumentParser(description="Grid search tif postprocess params.")
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--pred-root", required=True)
    parser.add_argument("--region", required=True)
    parser.add_argument("--stems", nargs="+", required=True)
    parser.add_argument("--output-csv", default="postprocess_param_sweep.csv")
    parser.add_argument("--prob-thresholds", default="0.08,0.10,0.12,0.14,0.16,0.18")
    parser.add_argument("--candidate-prob-thresholds", default="0.03,0.04,0.05,0.06,0.07,0.08,0.10")
    parser.add_argument("--overlap-thresholds", default="0.35,0.45,0.55,0.65")
    parser.add_argument("--min-object-areas", default="1,10,20,40")
    parser.add_argument("--min-component-areas", default="1,10,20,40,80")
    parser.add_argument("--max-hole-areas", default="0,32,64,128,256")
    parser.add_argument("--nodata", type=int, default=IGNORE_LABEL)
    return parser.parse_args()


def main():
    args = parse_args()
    samples = load_samples(args)
    rows = []

    grids = {
        "prob_threshold": parse_list(args.prob_thresholds, float),
        "candidate_prob_threshold": parse_list(args.candidate_prob_thresholds, float),
        "overlap_threshold": parse_list(args.overlap_thresholds, float),
        "min_object_area": parse_list(args.min_object_areas, int),
        "min_component_area": parse_list(args.min_component_areas, int),
        "max_hole_area": parse_list(args.max_hole_areas, int),
    }

    keys = list(grids.keys())
    total = np.prod([len(grids[key]) for key in keys])
    print(f"[INFO] Evaluating {total} parameter combinations")

    for values in product(*(grids[key] for key in keys)):
        params = dict(zip(keys, values))
        sessrs_metrics, post_metrics = evaluate_variant(args, params, samples)
        for stage, metrics in (("sessrs", sessrs_metrics), ("post", post_metrics)):
            cm = np.asarray(metrics_from_cm.__globals__.get("cm", np.zeros((2, 2))))
            row = {
                **params,
                "stage": stage,
                "pixels": metrics["pixels"],
                "oa": metrics["oa"],
                "target_recall": metrics["class_acc"][1],
                "target_f1": metrics["f1"][1],
                "background_iou": metrics["iou"][0],
                "target_iou": metrics["iou"][1],
                "miou": metrics["miou"],
            }
            rows.append(row)

    rows.sort(key=lambda row: (row["miou"], row["target_iou"]), reverse=True)
    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("[INFO] Top 10:")
    for row in rows[:10]:
        print(
            f"{row['stage']} miou={row['miou']:.4f} target_iou={row['target_iou']:.4f} "
            f"f1={row['target_f1']:.4f} recall={row['target_recall']:.4f} "
            f"prob={row['prob_threshold']} cand={row['candidate_prob_threshold']} "
            f"overlap={row['overlap_threshold']} min_obj={row['min_object_area']} "
            f"min_comp={row['min_component_area']} hole={row['max_hole_area']}"
        )
    print(f"[INFO] Saved: {out_path}")


if __name__ == "__main__":
    main()

import argparse
import csv
import json
import random
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

from batch_eval_tif_postprocess import find_mask, row_from_metrics
from batch_eval_ndvi_postprocess import compute_ndvi_stack
from test_manual_tiles import (
    DEFAULT_DATA_ROOT,
    IGNORE_LABEL,
    confusion_matrix,
    load_sample,
    metrics_from_cm,
)
from tif_binary_postprocess import (
    connected_components,
    fill_small_holes,
    make_candidate_object_labels,
    read_single_band,
    remove_small_components,
    write_single_band,
)


FEATURE_NAMES = [
    "region_id",
    "area",
    "bbox_w",
    "bbox_h",
    "aspect",
    "fill_ratio",
    "pred_overlap",
    "mean_prob",
    "max_prob",
    "std_prob",
    "p90_prob",
    "ndvi_max",
    "ndvi_mean",
    "ndvi_min",
    "ndvi_amp",
    "ndvi_peak_idx",
    "ndvi_t1",
    "ndvi_t2",
    "ndvi_t3",
    "ndvi_t4",
    "ndvi_t5",
    "ndvi_t6",
]


def collect_samples(pred_root, regions, seed, train_ratio):
    samples = []
    pred_root = Path(pred_root)
    for region in regions:
        pred_dir = pred_root / region / "pred"
        for pred_path in sorted(pred_dir.glob("*_pred.tif")):
            stem = pred_path.name[:-9]
            samples.append({"region": region, "stem": stem})
    rng = random.Random(seed)
    rng.shuffle(samples)
    split = int(round(len(samples) * train_ratio))
    return samples[:split], samples[split:]


def object_features(region_id, obj, pred, prob, ndvi_stack):
    ys, xs = np.nonzero(obj)
    area = int(ys.size)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    bbox_h = y1 - y0
    bbox_w = x1 - x0
    fill_ratio = area / float(max(bbox_h * bbox_w, 1))
    aspect = bbox_w / float(max(bbox_h, 1))

    prob_values = prob[obj]
    pred_overlap = float(np.count_nonzero(pred & obj)) / float(max(area, 1))

    ndvi_values = ndvi_stack[:, obj]
    if ndvi_values.size == 0:
        ndvi_series = np.zeros(6, dtype=np.float32)
    else:
        ndvi_series = np.nanmean(ndvi_values, axis=1).astype(np.float32)
    ndvi_max = float(np.nanmax(ndvi_series))
    ndvi_min = float(np.nanmin(ndvi_series))
    ndvi_mean = float(np.nanmean(ndvi_series))
    ndvi_amp = ndvi_max - ndvi_min
    ndvi_peak_idx = int(np.nanargmax(ndvi_series))

    return [
        float(region_id),
        float(area),
        float(bbox_w),
        float(bbox_h),
        float(aspect),
        float(fill_ratio),
        float(pred_overlap),
        float(np.mean(prob_values)),
        float(np.max(prob_values)),
        float(np.std(prob_values)),
        float(np.percentile(prob_values, 90)),
        ndvi_max,
        ndvi_mean,
        ndvi_min,
        ndvi_amp,
        float(ndvi_peak_idx),
        *[float(v) for v in ndvi_series],
    ]


def load_arrays(args, sample):
    region = sample["region"]
    stem = sample["stem"]
    region_dir = Path(args.data_root) / region
    pred_path = Path(args.pred_root) / region / "pred" / f"{stem}_pred.tif"
    prob_path = Path(args.pred_root) / region / "prob" / f"{stem}_prob.tif"
    mask_path = find_mask(args.data_root, region, stem)

    pred_arr, profile = read_single_band(pred_path)
    prob, _ = read_single_band(prob_path)
    gt, _ = read_single_band(mask_path)
    image, _ = load_sample(region_dir, f"{stem}.tif", ignore_nodata=True)

    gt = (gt > 0).astype(np.uint8)
    gt[pred_arr == args.nodata] = args.nodata
    prob = prob.astype(np.float32)
    if prob.max() > 1.0:
        prob = prob / 255.0
    pred = pred_arr == 1
    ndvi_stack = compute_ndvi_stack(image, args.red_band, args.nir_band)
    return pred_arr, pred, prob, gt, ndvi_stack, profile


def extract_objects(args, samples, region_to_id):
    rows = []
    labels = []
    meta = []
    for sample in samples:
        pred_arr, pred, prob, gt, ndvi_stack, _ = load_arrays(args, sample)
        candidate_labels = make_candidate_object_labels(pred, prob, args.candidate_prob_threshold)
        for obj_id in np.unique(candidate_labels):
            if obj_id == 0:
                continue
            obj = candidate_labels == obj_id
            area = int(np.count_nonzero(obj))
            if area < args.min_object_area:
                continue
            valid = gt[obj] != args.nodata
            if not np.any(valid):
                continue
            target_ratio = float(np.mean(gt[obj][valid] == 1))
            label = 1 if target_ratio >= args.positive_target_ratio else 0
            rows.append(object_features(region_to_id[sample["region"]], obj, pred, prob, ndvi_stack))
            labels.append(label)
            meta.append({
                "region": sample["region"],
                "stem": sample["stem"],
                "object_id": int(obj_id),
                "target_ratio": target_ratio,
                "area": area,
            })
    return np.asarray(rows, dtype=np.float32), np.asarray(labels, dtype=np.uint8), meta


def predict_sample(args, clf, sample, region_to_id):
    pred_arr, pred, prob, gt, ndvi_stack, profile = load_arrays(args, sample)
    candidate_labels = make_candidate_object_labels(pred, prob, args.candidate_prob_threshold)
    refined = np.zeros(pred.shape, dtype=bool)

    for obj_id in np.unique(candidate_labels):
        if obj_id == 0:
            continue
        obj = candidate_labels == obj_id
        if np.count_nonzero(obj) < args.min_object_area:
            continue
        feat = np.asarray([object_features(region_to_id[sample["region"]], obj, pred, prob, ndvi_stack)], dtype=np.float32)
        keep_prob = float(clf.predict_proba(feat)[0, 1])
        if keep_prob >= args.rf_threshold:
            refined[obj] = True

    refined = remove_small_components(refined, args.min_component_area)
    refined = fill_small_holes(refined, args.max_hole_area)
    out = np.full(pred_arr.shape, args.nodata, dtype=np.uint8)
    valid = pred_arr != args.nodata
    out[valid] = refined.astype(np.uint8)[valid]
    return pred_arr, out, gt, profile


def evaluate_split(args, clf, samples, region_to_id, split_name):
    out_root = Path(args.out_root) / split_name
    out_root.mkdir(parents=True, exist_ok=True)
    overall_before = np.zeros((2, 2), dtype=np.int64)
    overall_after = np.zeros((2, 2), dtype=np.int64)
    rows = []

    for sample in samples:
        pred_arr, out, gt, profile = predict_sample(args, clf, sample, region_to_id)
        region_out = out_root / sample["region"]
        out_path = region_out / f"{sample['stem']}_pred_rf_post.tif"
        write_single_band(out_path, out, profile, dtype="uint8", nodata=args.nodata)

        cm_before = confusion_matrix(pred_arr, gt)
        cm_after = confusion_matrix(out, gt)
        overall_before += cm_before
        overall_after += cm_after

        for stage, cm in (("before", cm_before), ("rf_after", cm_after)):
            metrics = metrics_from_cm(cm)
            metrics["confusion_matrix"] = cm.tolist()
            row = row_from_metrics(f"{sample['region']}/{sample['stem']}", stage, metrics)
            row["split"] = split_name
            rows.append(row)

    for stage, cm in (("before", overall_before), ("rf_after", overall_after)):
        metrics = metrics_from_cm(cm)
        metrics["confusion_matrix"] = cm.tolist()
        row = row_from_metrics("overall", stage, metrics)
        row["split"] = split_name
        rows.append(row)

    return rows, overall_before, overall_after


def parse_args():
    parser = argparse.ArgumentParser(description="Train object-level RandomForest postprocessor.")
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--pred-root", required=True)
    parser.add_argument("--out-root", default="object_rf_postprocess")
    parser.add_argument("--regions", nargs="+", default=["滨城区3镇", "阳信县4镇"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--red-band", type=int, default=2)
    parser.add_argument("--nir-band", type=int, default=3)
    parser.add_argument("--candidate-prob-threshold", type=float, default=0.04)
    parser.add_argument("--positive-target-ratio", type=float, default=0.50)
    parser.add_argument("--rf-threshold", type=float, default=0.50)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=12)
    parser.add_argument("--min-object-area", type=int, default=20)
    parser.add_argument("--min-component-area", type=int, default=20)
    parser.add_argument("--max-hole-area", type=int, default=64)
    parser.add_argument("--nodata", type=int, default=IGNORE_LABEL)
    return parser.parse_args()


def main():
    args = parse_args()
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    region_to_id = {region: idx for idx, region in enumerate(args.regions)}

    train_samples, val_samples = collect_samples(args.pred_root, args.regions, args.seed, args.train_ratio)
    print(f"[INFO] Samples: train={len(train_samples)}, val={len(val_samples)}")

    x_train, y_train, train_meta = extract_objects(args, train_samples, region_to_id)
    x_val, y_val, val_meta = extract_objects(args, val_samples, region_to_id)
    print(f"[INFO] Objects: train={len(y_train)} pos={int(y_train.sum())}, val={len(y_val)} pos={int(y_val.sum())}")

    clf = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        class_weight="balanced_subsample",
        random_state=args.seed,
        n_jobs=-1,
    )
    clf.fit(x_train, y_train)

    print("[INFO] Object-level validation report:")
    print(classification_report(y_val, clf.predict(x_val), digits=4))

    joblib.dump({"model": clf, "feature_names": FEATURE_NAMES, "args": vars(args)}, out_root / "object_rf.joblib")
    with (out_root / "feature_importances.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["feature", "importance"])
        for name, importance in sorted(zip(FEATURE_NAMES, clf.feature_importances_), key=lambda x: x[1], reverse=True):
            writer.writerow([name, float(importance)])

    all_rows = []
    summary = {}
    for split_name, samples in (("train", train_samples), ("val", val_samples)):
        rows, before_cm, after_cm = evaluate_split(args, clf, samples, region_to_id, split_name)
        all_rows.extend(rows)
        before = metrics_from_cm(before_cm)
        after = metrics_from_cm(after_cm)
        summary[split_name] = {
            "before": {**before, "confusion_matrix": before_cm.tolist()},
            "rf_after": {**after, "confusion_matrix": after_cm.tolist()},
        }
        print(
            f"[{split_name}] MIoU {before['miou']:.4f} -> {after['miou']:.4f}, "
            f"target IoU {before['iou'][1]:.4f} -> {after['iou'][1]:.4f}, "
            f"target F1 {before['f1'][1]:.4f} -> {after['f1'][1]:.4f}"
        )

    metrics_csv = out_root / "object_rf_metrics.csv"
    with metrics_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    with (out_root / "object_rf_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[INFO] Saved outputs: {out_root}")


if __name__ == "__main__":
    main()

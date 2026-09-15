"""Tune constrained temporal-frequency parcel splitting to reduce under-segmentation."""

import argparse
import csv
import json
from itertools import product
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from evaluate_parcel_metrics import evaluate_case
from make_frequency_best_result_figures import read_prob
from test_manual_tiles import IGNORE_LABEL, confusion_matrix, load_sample, metrics_from_cm
from tif_binary_postprocess import write_single_band


def normalized_corridor_scores(image):
    data = image.reshape(6, 6, image.shape[-2], image.shape[-1]).astype(np.float32)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    scores = []
    for period in (2, 3, 4):
        red, nir = data[period, 2], data[period, 3]
        ndvi = (nir - red) / np.maximum(nir + red, 1e-6)
        smooth = cv2.GaussianBlur(ndvi, (3, 3), 0)
        score = np.maximum(cv2.morphologyEx(smooth, cv2.MORPH_CLOSE, close_kernel) - smooth, 0)
        lo, hi = np.percentile(score, [2, 98])
        scores.append(np.clip((score - lo) / max(hi - lo, 1e-6), 0, 1))
    return np.stack(scores, axis=0)


def large_component_interior(mask, minimum_area, margin=1):
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    interior = np.zeros_like(mask, dtype=bool)
    objects = []
    kernel = np.ones((margin * 2 + 1, margin * 2 + 1), dtype=np.uint8)
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < minimum_area:
            continue
        obj = labels == label
        interior |= cv2.erode(obj.astype(np.uint8), kernel) > 0
        objects.append(obj)
    return interior, objects


def separator_candidates(scores, interior, quantile, minimum_votes, line_length, minimum_length):
    votes = np.zeros(interior.shape, dtype=np.uint8)
    for score in scores:
        values = score[interior]
        threshold = float(np.quantile(values, quantile)) if values.size else 1.0
        votes += (score >= threshold).astype(np.uint8)
    candidate = interior & (votes >= minimum_votes)
    horizontal = cv2.morphologyEx(
        candidate.astype(np.uint8), cv2.MORPH_OPEN, np.ones((1, line_length), np.uint8)
    )
    vertical = cv2.morphologyEx(
        candidate.astype(np.uint8), cv2.MORPH_OPEN, np.ones((line_length, 1), np.uint8)
    )
    lines = (horizontal > 0) | (vertical > 0)
    lines = cv2.morphologyEx(lines.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)) > 0

    count, labels, stats, _ = cv2.connectedComponentsWithStats(lines.astype(np.uint8), 8)
    kept = np.zeros_like(lines)
    for label in range(1, count):
        x, y, width, height, area = stats[label]
        if area >= minimum_length and max(width, height) >= minimum_length:
            kept[labels == label] = True
    return kept & interior


def split_with_guard(initial, probability, scores, params, minimum_piece_area=20):
    interior, objects = large_component_interior(initial, params["minimum_object_area"])
    lines = separator_candidates(
        scores,
        interior,
        params["quantile"],
        params["minimum_votes"],
        params["line_length"],
        params["minimum_length"],
    )
    carve = lines & (probability < params["carve_probability"])
    if params["dilate_width"] > 1:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (params["dilate_width"], params["dilate_width"])
        )
        carve = cv2.dilate(carve.astype(np.uint8), kernel) > 0
        carve &= interior

    accepted = np.zeros_like(initial)
    split_objects = 0
    for obj in objects:
        local_carve = carve & obj
        if not np.any(local_carve):
            continue
        trial = obj & ~local_carve
        count, _, stats, _ = cv2.connectedComponentsWithStats(trial.astype(np.uint8), 8)
        pieces = stats[1:, cv2.CC_STAT_AREA] if count > 1 else np.asarray([])
        reasonable = pieces[pieces >= minimum_piece_area]
        retained_fraction = reasonable.sum() / max(int(obj.sum()), 1)
        if len(reasonable) >= 2 and retained_fraction >= 0.92:
            accepted |= local_carve
            split_objects += 1

    refined = initial.copy()
    refined[accepted] = False
    return refined, accepted, split_objects


def aggregate(cases, params, threshold, min_parcel_area, pixel_size):
    cm = np.zeros((2, 2), dtype=np.int64)
    totals = {key: 0 for key in ("gt_parcels", "pred_parcels", "matched_parcels", "over_segmented", "under_segmented")}
    area_errors, offsets = [], []
    carved_pixels = 0
    split_objects = 0
    predictions = []
    for case in cases:
        initial = (case["probability"] >= threshold) & case["valid"]
        if params is None:
            refined, carved, split_count = initial, np.zeros_like(initial), 0
        else:
            refined, carved, split_count = split_with_guard(
                initial, case["probability"], case["scores"], params, min_parcel_area
            )
        result = evaluate_case(
            case["gt"], refined, min_parcel_area, 0.50, 0.10, pixel_size
        )
        for key in totals:
            totals[key] += result[key]
        area_errors.extend(result["area_errors"])
        offsets.extend(result["centroid_offsets_m"])
        out = np.full(case["label"].shape, IGNORE_LABEL, dtype=np.uint8)
        out[case["valid"]] = refined[case["valid"]].astype(np.uint8)
        cm += confusion_matrix(out, case["label"])
        carved_pixels += int(carved.sum())
        split_objects += split_count
        predictions.append((refined, carved))

    matched = totals["matched_parcels"]
    precision = matched / max(totals["pred_parcels"], 1)
    recall = matched / max(totals["gt_parcels"], 1)
    pixel = metrics_from_cm(cm)
    metrics = {
        **totals,
        "parcel_precision": precision,
        "parcel_recall": recall,
        "parcel_f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "over_segmentation_rate": totals["over_segmented"] / max(totals["gt_parcels"], 1),
        "under_segmentation_rate": totals["under_segmented"] / max(totals["pred_parcels"], 1),
        "mean_area_error_percent": float(np.mean(area_errors) * 100) if area_errors else None,
        "mean_centroid_offset_m": float(np.mean(offsets)) if offsets else None,
        "target_iou": pixel["iou"][1],
        "miou": pixel["miou"],
        "carved_pixels": carved_pixels,
        "split_objects": split_objects,
    }
    return metrics, predictions


def load_cases(args):
    split = json.loads(Path(args.split_file).read_text(encoding="utf-8"))
    items = [item for item in split[args.split_part] if item["region"] in args.regions]
    cases = []
    for item in tqdm(items, desc="Loading parcel cases"):
        region, sample = item["region"], item["sample"]
        image, label = load_sample(Path(args.data_root) / region, sample, ignore_nodata=True)
        stem = Path(sample).stem
        probability = read_prob(Path(args.prob_root) / region / "prob" / f"{stem}_prob.tif")
        cases.append(
            {
                "region": region,
                "sample": sample,
                "image": image,
                "label": label,
                "valid": label != IGNORE_LABEL,
                "gt": label == 1,
                "probability": probability,
                "scores": normalized_corridor_scores(image),
            }
        )
    return cases


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
    args = parser.parse_args()

    cases = load_cases(args)
    baseline, _ = aggregate(cases, None, args.threshold, args.min_parcel_area, args.pixel_size)
    rows = []
    if args.params_json:
        params_list = [json.loads(Path(args.params_json).read_text(encoding="utf-8"))["best_params"]]
    else:
        params_list = [
            {
                "quantile": quantile,
                "minimum_votes": votes,
                "line_length": line_length,
                "minimum_length": min_length,
                "minimum_object_area": object_area,
                "carve_probability": carve_probability,
                "dilate_width": 1,
            }
            for quantile, votes, line_length, min_length, object_area, carve_probability in product(
                [0.60, 0.70, 0.80], [1, 2], [3, 5], [6, 12], [300, 600], [0.80, 0.95]
            )
        ]

    for params in tqdm(params_list, desc="Parcel split search"):
        metrics, _ = aggregate(cases, params, args.threshold, args.min_parcel_area, args.pixel_size)
        pixel_drop = baseline["target_iou"] - metrics["target_iou"]
        over_increase = max(metrics["over_segmentation_rate"] - baseline["over_segmentation_rate"], 0.0)
        score = (
            metrics["parcel_f1"]
            - 0.30 * metrics["under_segmentation_rate"]
            - 0.50 * over_increase
            - 2.0 * max(pixel_drop - 0.015, 0.0)
        )
        rows.append({**params, **metrics, "selection_score": score})
    rows.sort(key=lambda item: item["selection_score"], reverse=True)
    best_params = {key: rows[0][key] for key in params_list[0]}
    best_metrics, predictions = aggregate(
        cases, best_params, args.threshold, args.min_parcel_area, args.pixel_size
    )

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "parcel_split_sweep.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    if args.save_best:
        for case, (prediction, carved) in zip(cases, predictions):
            output_mask = np.full(case["label"].shape, IGNORE_LABEL, dtype=np.uint8)
            output_mask[case["valid"]] = prediction[case["valid"]].astype(np.uint8)
            stem = Path(case["sample"]).stem
            profile_path = Path(args.data_root) / case["region"] / "mask" / case["sample"]
            import rasterio

            with rasterio.open(profile_path) as src:
                profile = src.profile.copy()
            write_single_band(
                output / case["region"] / "pred" / f"{stem}_pred.tif",
                output_mask,
                profile,
            )

    summary = {
        "samples": len(cases),
        "split_part": args.split_part,
        "regions": args.regions,
        "baseline": baseline,
        "best_params": best_params,
        "best": best_metrics,
    }
    (output / "parcel_split_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

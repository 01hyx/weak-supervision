import argparse
import csv
from pathlib import Path

import numpy as np

from batch_eval_tif_postprocess import find_mask, row_from_metrics
from test_manual_tiles import (
    DEFAULT_DATA_ROOT,
    IGNORE_LABEL,
    TIME_STEPS,
    BANDS_PER_STEP,
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


def compute_ndvi_stack(image_chw, red_band, nir_band):
    h, w = image_chw.shape[-2:]
    data_t = image_chw.reshape(TIME_STEPS, BANDS_PER_STEP, h, w)
    red = data_t[:, red_band].astype(np.float32)
    nir = data_t[:, nir_band].astype(np.float32)
    return (nir - red) / np.maximum(nir + red, 1e-6)


def object_ndvi_stats(ndvi_stack, mask):
    values = ndvi_stack[:, mask]
    if values.size == 0:
        return 0.0, 0.0, 0.0
    series = np.nanmean(values, axis=1)
    ndvi_max = float(np.nanmax(series))
    ndvi_mean = float(np.nanmean(series))
    ndvi_amp = float(np.nanmax(series) - np.nanmin(series))
    return ndvi_max, ndvi_mean, ndvi_amp


def ndvi_refine(pred, prob, ndvi_stack, args):
    candidate_labels = make_candidate_object_labels(pred, prob, args.candidate_prob_threshold)
    object_ids = [obj_id for obj_id in np.unique(candidate_labels) if obj_id != 0]
    refined = np.zeros_like(pred, dtype=bool)

    for obj_id in object_ids:
        obj = candidate_labels == obj_id
        area = int(np.count_nonzero(obj))
        if area < args.min_object_area:
            continue

        overlap = float(np.count_nonzero(pred & obj)) / float(area)
        mean_prob = float(np.mean(prob[obj]))
        ndvi_max, ndvi_mean, ndvi_amp = object_ndvi_stats(ndvi_stack, obj)

        prob_accept = mean_prob >= args.prob_threshold or overlap >= args.overlap_threshold
        ndvi_accept = (
            ndvi_max >= args.min_ndvi_max
            and ndvi_mean >= args.min_ndvi_mean
            and ndvi_amp >= args.min_ndvi_amp
        )

        if prob_accept and ndvi_accept:
            refined[obj] = True

    # Keep high-confidence existing predictions unless their NDVI is clearly weak.
    pred_labels, pred_components = connected_components(pred)
    for idx, coords in enumerate(pred_components, start=1):
        obj = pred_labels == idx
        if coords.shape[0] < args.min_component_area:
            continue
        ndvi_max, ndvi_mean, _ = object_ndvi_stats(ndvi_stack, obj)
        mean_prob = float(np.mean(prob[obj]))
        if mean_prob >= args.keep_prob_threshold and ndvi_max >= args.keep_min_ndvi_max and ndvi_mean >= args.keep_min_ndvi_mean:
            refined[obj] = True

    refined = remove_small_components(refined, args.min_component_area)
    refined = fill_small_holes(refined, args.max_hole_area)
    return refined.astype(np.uint8)


def parse_args():
    parser = argparse.ArgumentParser(description="NDVI-aware binary postprocess and evaluation.")
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--pred-root", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--region", required=True)
    parser.add_argument("--all-stems", action="store_true")
    parser.add_argument("--stems", nargs="+", default=None)
    parser.add_argument("--red-band", type=int, default=2)
    parser.add_argument("--nir-band", type=int, default=3)
    parser.add_argument("--prob-threshold", type=float, default=0.10)
    parser.add_argument("--candidate-prob-threshold", type=float, default=0.04)
    parser.add_argument("--overlap-threshold", type=float, default=0.35)
    parser.add_argument("--min-ndvi-max", type=float, default=0.20)
    parser.add_argument("--min-ndvi-mean", type=float, default=0.05)
    parser.add_argument("--min-ndvi-amp", type=float, default=0.03)
    parser.add_argument("--keep-prob-threshold", type=float, default=0.30)
    parser.add_argument("--keep-min-ndvi-max", type=float, default=0.10)
    parser.add_argument("--keep-min-ndvi-mean", type=float, default=0.00)
    parser.add_argument("--min-object-area", type=int, default=20)
    parser.add_argument("--min-component-area", type=int, default=20)
    parser.add_argument("--max-hole-area", type=int, default=64)
    parser.add_argument("--nodata", type=int, default=IGNORE_LABEL)
    return parser.parse_args()


def main():
    args = parse_args()
    pred_root = Path(args.pred_root) / args.region
    out_root = Path(args.out_root) / args.region
    out_root.mkdir(parents=True, exist_ok=True)

    if args.all_stems:
        args.stems = sorted(path.name[:-9] for path in (pred_root / "pred").glob("*_pred.tif"))
    if not args.stems:
        raise ValueError("Provide --stems or --all-stems.")

    rows = []
    overall_before = np.zeros((2, 2), dtype=np.int64)
    overall_after = np.zeros((2, 2), dtype=np.int64)

    for stem in args.stems:
        pred_path = pred_root / "pred" / f"{stem}_pred.tif"
        prob_path = pred_root / "prob" / f"{stem}_prob.tif"
        mask_path = find_mask(args.data_root, args.region, stem)
        sample_name = f"{stem}.tif"

        pred_arr, profile = read_single_band(pred_path)
        prob, _ = read_single_band(prob_path)
        gt, _ = read_single_band(mask_path)
        image, _ = load_sample(Path(args.data_root) / args.region, sample_name, ignore_nodata=True)

        gt = (gt > 0).astype(np.int64)
        gt[pred_arr == args.nodata] = args.nodata
        prob = prob.astype(np.float32)
        if prob.max() > 1.0:
            prob = prob / 255.0

        pred = pred_arr == 1
        ndvi_stack = compute_ndvi_stack(image, args.red_band, args.nir_band)
        refined = ndvi_refine(pred, prob, ndvi_stack, args)

        out = np.full(pred_arr.shape, args.nodata, dtype=np.uint8)
        valid = pred_arr != args.nodata
        out[valid] = refined[valid]
        out_path = out_root / f"{stem}_pred_ndvi_post.tif"
        write_single_band(out_path, out, profile, dtype="uint8", nodata=args.nodata)

        cm_before = confusion_matrix(pred_arr, gt)
        cm_after = confusion_matrix(out, gt)
        overall_before += cm_before
        overall_after += cm_after

        before = metrics_from_cm(cm_before)
        before["confusion_matrix"] = cm_before.tolist()
        after = metrics_from_cm(cm_after)
        after["confusion_matrix"] = cm_after.tolist()
        rows.append(row_from_metrics(stem, "before", before))
        rows.append(row_from_metrics(stem, "ndvi_after", after))

    before = metrics_from_cm(overall_before)
    before["confusion_matrix"] = overall_before.tolist()
    after = metrics_from_cm(overall_after)
    after["confusion_matrix"] = overall_after.tolist()
    rows.append(row_from_metrics("overall", "before", before))
    rows.append(row_from_metrics("overall", "ndvi_after", after))

    csv_path = Path(args.out_root) / f"{args.region}_ndvi_postprocess_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(
        f"{args.region}: MIoU {before['miou']:.4f} -> {after['miou']:.4f}, "
        f"target IoU {before['iou'][1]:.4f} -> {after['iou'][1]:.4f}, "
        f"target F1 {before['f1'][1]:.4f} -> {after['f1'][1]:.4f}"
    )
    print(f"[INFO] Saved metrics: {csv_path}")


if __name__ == "__main__":
    main()

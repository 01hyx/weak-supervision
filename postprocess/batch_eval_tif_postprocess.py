import argparse
import csv
from pathlib import Path

import numpy as np

from test_manual_tiles import DEFAULT_DATA_ROOT, IGNORE_LABEL, confusion_matrix, metrics_from_cm
from tif_binary_postprocess import postprocess, read_single_band, sessrs_object_refine, write_single_band


def find_mask(data_root, region, stem):
    mask_path = Path(data_root) / region / "mask" / f"{stem}.tif"
    if mask_path.exists():
        return mask_path
    mask_path = Path(data_root) / region / "mask" / f"{stem}.tiff"
    if mask_path.exists():
        return mask_path
    raise FileNotFoundError(f"Mask not found for {region}/{stem}")


def evaluate_pair(pred, gt):
    cm = confusion_matrix(pred.astype(np.uint8), gt.astype(np.int64))
    metrics = metrics_from_cm(cm)
    metrics["confusion_matrix"] = cm.tolist()
    return metrics


def row_from_metrics(sample, stage, metrics):
    cm = np.asarray(metrics["confusion_matrix"])
    tn, fp = cm[0]
    fn, tp = cm[1]
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    return {
        "sample": sample,
        "stage": stage,
        "pixels": metrics["pixels"],
        "oa": metrics["oa"],
        "target_recall": metrics["class_acc"][1],
        "target_precision": precision,
        "target_f1": metrics["f1"][1],
        "background_iou": metrics["iou"][0],
        "target_iou": metrics["iou"][1],
        "miou": metrics["miou"],
        "fp": int(fp),
        "fn": int(fn),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Batch postprocess prediction GeoTIFFs and evaluate before/after.")
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--pred-root", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--region", required=True)
    parser.add_argument("--stems", nargs="+", default=None)
    parser.add_argument("--all-stems", action="store_true")
    parser.add_argument("--prob-threshold", type=float, default=0.12)
    parser.add_argument("--candidate-prob-threshold", type=float, default=0.06)
    parser.add_argument("--overlap-threshold", type=float, default=0.55)
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

        pred_arr, profile = read_single_band(pred_path)
        prob, _ = read_single_band(prob_path)
        gt, _ = read_single_band(mask_path)
        gt = (gt > 0).astype(np.int64)
        gt[pred_arr == args.nodata] = args.nodata

        pred = pred_arr == 1
        prob = prob.astype(np.float32)
        if prob.max() > 1.0:
            prob = prob / 255.0

        sessrs_refined = sessrs_object_refine(pred, prob, None, args).astype(np.uint8)
        sessrs_out = np.full(pred_arr.shape, args.nodata, dtype=np.uint8)
        valid = pred_arr != args.nodata
        sessrs_out[valid] = sessrs_refined[valid]
        sessrs_path = out_root / f"{stem}_pred_sessrs.tif"
        write_single_band(sessrs_path, sessrs_out, profile, dtype="uint8", nodata=args.nodata)

        refined = postprocess(pred, prob, None, args)
        out = np.full(pred_arr.shape, args.nodata, dtype=np.uint8)
        out[valid] = refined[valid]
        out_path = out_root / f"{stem}_pred_post.tif"
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
        rows.append(row_from_metrics(stem, "after", after))

        print(
            f"{stem}: MIoU {before['miou']:.4f} -> {after['miou']:.4f}, "
            f"target IoU {before['iou'][1]:.4f} -> {after['iou'][1]:.4f}"
        )

    before = metrics_from_cm(overall_before)
    before["confusion_matrix"] = overall_before.tolist()
    after = metrics_from_cm(overall_after)
    after["confusion_matrix"] = overall_after.tolist()
    rows.append(row_from_metrics("overall", "before", before))
    rows.append(row_from_metrics("overall", "after", after))

    csv_path = Path(args.out_root) / f"{args.region}_postprocess_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(
        f"overall: MIoU {before['miou']:.4f} -> {after['miou']:.4f}, "
        f"target IoU {before['iou'][1]:.4f} -> {after['iou'][1]:.4f}, "
        f"target F1 {before['f1'][1]:.4f} -> {after['f1'][1]:.4f}"
    )
    print(f"[INFO] Saved metrics: {csv_path}")


if __name__ == "__main__":
    main()

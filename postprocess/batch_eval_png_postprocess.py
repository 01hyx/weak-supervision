import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image

from batch_eval_tif_postprocess import find_mask, row_from_metrics
from test_manual_tiles import DEFAULT_DATA_ROOT, IGNORE_LABEL, confusion_matrix, metrics_from_cm
from tif_binary_postprocess import postprocess, read_single_band, sessrs_object_refine


def save_png(path, arr):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr.astype(np.uint8)).save(path)


def read_png(path):
    return np.array(Image.open(path))


def parse_args():
    parser = argparse.ArgumentParser(description="Convert prediction/probability tif to PNG, postprocess PNG arrays, and evaluate.")
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--pred-root", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--region", required=True)
    parser.add_argument("--all-stems", action="store_true")
    parser.add_argument("--stems", nargs="+", default=None)
    parser.add_argument("--prob-threshold", type=float, default=0.10)
    parser.add_argument("--candidate-prob-threshold", type=float, default=0.04)
    parser.add_argument("--overlap-threshold", type=float, default=0.35)
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
    overall_sessrs = np.zeros((2, 2), dtype=np.int64)
    overall_after = np.zeros((2, 2), dtype=np.int64)

    for stem in args.stems:
        pred_tif = pred_root / "pred" / f"{stem}_pred.tif"
        prob_tif = pred_root / "prob" / f"{stem}_prob.tif"
        mask_tif = find_mask(args.data_root, args.region, stem)

        pred_arr, _ = read_single_band(pred_tif)
        prob_arr, _ = read_single_band(prob_tif)
        gt, _ = read_single_band(mask_tif)
        gt = (gt > 0).astype(np.uint8)
        gt[pred_arr == args.nodata] = args.nodata

        prob_float = prob_arr.astype(np.float32)
        if prob_float.max() > 1.0:
            prob_float = prob_float / 255.0
        prob_png = np.clip(np.rint(prob_float * 255.0), 0, 255).astype(np.uint8)
        pred_png = pred_arr.astype(np.uint8)

        png_dir = out_root / "png_inputs"
        save_png(png_dir / "pred" / f"{stem}_pred.png", pred_png)
        save_png(png_dir / "prob" / f"{stem}_prob.png", prob_png)
        save_png(png_dir / "mask" / f"{stem}_mask.png", gt)

        pred_png = read_png(png_dir / "pred" / f"{stem}_pred.png")
        prob = read_png(png_dir / "prob" / f"{stem}_prob.png").astype(np.float32) / 255.0
        gt_png = read_png(png_dir / "mask" / f"{stem}_mask.png")

        pred_bool = pred_png == 1
        valid = pred_png != args.nodata

        sessrs = sessrs_object_refine(pred_bool, prob, None, args).astype(np.uint8)
        sessrs_out = np.full(pred_png.shape, args.nodata, dtype=np.uint8)
        sessrs_out[valid] = sessrs[valid]

        post = postprocess(pred_bool, prob, None, args).astype(np.uint8)
        post_out = np.full(pred_png.shape, args.nodata, dtype=np.uint8)
        post_out[valid] = post[valid]

        save_png(out_root / "sessrs_png" / f"{stem}_pred_sessrs.png", sessrs_out)
        save_png(out_root / "post_png" / f"{stem}_pred_post.png", post_out)

        cm_before = confusion_matrix(pred_png, gt_png)
        cm_sessrs = confusion_matrix(sessrs_out, gt_png)
        cm_after = confusion_matrix(post_out, gt_png)
        overall_before += cm_before
        overall_sessrs += cm_sessrs
        overall_after += cm_after

        for stage, cm in (("before", cm_before), ("sessrs", cm_sessrs), ("after", cm_after)):
            metrics = metrics_from_cm(cm)
            metrics["confusion_matrix"] = cm.tolist()
            rows.append(row_from_metrics(stem, stage, metrics))

    for stage, cm in (("before", overall_before), ("sessrs", overall_sessrs), ("after", overall_after)):
        metrics = metrics_from_cm(cm)
        metrics["confusion_matrix"] = cm.tolist()
        rows.append(row_from_metrics("overall", stage, metrics))

    csv_path = Path(args.out_root) / f"{args.region}_png_postprocess_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    before = metrics_from_cm(overall_before)
    sessrs = metrics_from_cm(overall_sessrs)
    after = metrics_from_cm(overall_after)
    print(
        f"{args.region}: MIoU {before['miou']:.4f} -> {sessrs['miou']:.4f} -> {after['miou']:.4f}, "
        f"target IoU {before['iou'][1]:.4f} -> {sessrs['iou'][1]:.4f} -> {after['iou'][1]:.4f}, "
        f"target F1 {before['f1'][1]:.4f} -> {sessrs['f1'][1]:.4f} -> {after['f1'][1]:.4f}"
    )
    print(f"[INFO] Saved metrics: {csv_path}")


if __name__ == "__main__":
    main()

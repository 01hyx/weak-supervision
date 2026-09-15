import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from test_manual_tiles import DEFAULT_DATA_ROOT, IGNORE_LABEL, collect_valid_sample_ids, discover_regions
from tif_binary_postprocess import read_single_band


def boundary_mask(mask, radius):
    kernel = np.ones((2 * radius + 1, 2 * radius + 1), dtype=np.uint8)
    mask_u8 = mask.astype(np.uint8)
    dilated = cv2.dilate(mask_u8, kernel)
    eroded = cv2.erode(mask_u8, kernel)
    return (dilated != eroded)


def safe_div(num, den):
    return float(num) / float(den) if den else 0.0


def score_case(gt, pred, valid, boundary_radius):
    gt_one = gt == 1
    pred_one = pred == 1

    tp = int(np.count_nonzero(valid & gt_one & pred_one))
    tn = int(np.count_nonzero(valid & ~gt_one & ~pred_one))
    fp = int(np.count_nonzero(valid & ~gt_one & pred_one))
    fn = int(np.count_nonzero(valid & gt_one & ~pred_one))
    target_pixels = int(np.count_nonzero(valid & gt_one))
    pred_pixels = int(np.count_nonzero(valid & pred_one))

    gt_boundary = boundary_mask(gt_one & valid, boundary_radius)
    pred_boundary = boundary_mask(pred_one & valid, boundary_radius)
    boundary_valid = valid & (gt_boundary | pred_boundary)
    boundary_errors = int(np.count_nonzero(boundary_valid & (gt_one != pred_one)))
    boundary_pixels = int(np.count_nonzero(boundary_valid))

    return {
        "pixels": int(np.count_nonzero(valid)),
        "target_pixels": target_pixels,
        "pred_pixels": pred_pixels,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "fn_rate": safe_div(fn, target_pixels),
        "fp_rate": safe_div(fp, int(np.count_nonzero(valid & ~gt_one))),
        "precision": safe_div(tp, tp + fp),
        "recall": safe_div(tp, tp + fn),
        "target_iou": safe_div(tp, tp + fp + fn),
        "boundary_pixels": boundary_pixels,
        "boundary_errors": boundary_errors,
        "boundary_error_rate": safe_div(boundary_errors, boundary_pixels),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Rank tiles by false negatives and boundary errors.")
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--pred-root", default="manual_tile_prediction_tifs_finetune_head")
    parser.add_argument("--regions", nargs="*", default=None)
    parser.add_argument("--threshold", type=float, default=0.70)
    parser.add_argument("--boundary-radius", type=int, default=2)
    parser.add_argument("--output-csv", default="manual_tile_error_rankings/finetune_head_error_ranking.csv")
    return parser.parse_args()


def main():
    args = parse_args()
    rows = []
    for region_dir in discover_regions(args.data_root, args.regions):
        region = region_dir.name
        for sample in tqdm(collect_valid_sample_ids(region_dir), desc=region):
            stem = Path(sample).stem
            prob_path = Path(args.pred_root) / region / "prob" / f"{stem}_prob.tif"
            pred_path = Path(args.pred_root) / region / "pred" / f"{stem}_pred.tif"
            mask_path = region_dir / "mask" / sample
            if not prob_path.exists() or not pred_path.exists():
                continue
            prob, _ = read_single_band(prob_path)
            pred_base, _ = read_single_band(pred_path)
            gt, _ = read_single_band(mask_path)

            prob = prob.astype(np.float32)
            if prob.max() > 1.0:
                prob = prob / 255.0
            valid = pred_base != IGNORE_LABEL
            gt = (gt > 0).astype(np.uint8)
            pred = (prob >= args.threshold).astype(np.uint8)
            pred[~valid] = 0

            metrics = score_case(gt, pred, valid, args.boundary_radius)
            rows.append({
                "region": region,
                "sample": sample,
                "stem": stem,
                "threshold": args.threshold,
                **metrics,
                "hard_score": metrics["fn_rate"] * 0.55 + metrics["boundary_error_rate"] * 0.45,
            })

    rows.sort(key=lambda item: item["hard_score"], reverse=True)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"[INFO] Saved: {output_csv}")
    print("\nTop false-negative tiles:")
    for row in sorted(rows, key=lambda item: item["fn"], reverse=True)[:10]:
        print(
            f"{row['region']}/{row['sample']} fn={row['fn']} fn_rate={row['fn_rate']:.3f} "
            f"boundary_error={row['boundary_error_rate']:.3f} target_iou={row['target_iou']:.3f}"
        )
    print("\nTop boundary-error tiles:")
    for row in sorted(rows, key=lambda item: item["boundary_error_rate"], reverse=True)[:10]:
        print(
            f"{row['region']}/{row['sample']} boundary_error={row['boundary_error_rate']:.3f} "
            f"fn={row['fn']} fn_rate={row['fn_rate']:.3f} target_iou={row['target_iou']:.3f}"
        )


if __name__ == "__main__":
    main()

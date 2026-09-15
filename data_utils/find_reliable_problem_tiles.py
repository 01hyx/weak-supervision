import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from test_manual_tiles import DEFAULT_DATA_ROOT, IGNORE_LABEL, collect_valid_sample_ids, discover_regions
from tif_binary_postprocess import read_single_band


def safe_div(num, den):
    return float(num) / float(den) if den else 0.0


def boundary_mask(mask, radius):
    kernel = np.ones((2 * radius + 1, 2 * radius + 1), dtype=np.uint8)
    mask_u8 = mask.astype(np.uint8)
    return cv2.dilate(mask_u8, kernel) != cv2.erode(mask_u8, kernel)


def make_inner_mask(shape, margin):
    h, w = shape
    inner = np.zeros((h, w), dtype=bool)
    inner[margin:h - margin, margin:w - margin] = True
    return inner


def score(gt, pred, valid, boundary_radius, edge_margin):
    gt_one = gt == 1
    pred_one = pred == 1
    fp_mask = valid & ~gt_one & pred_one
    fn_mask = valid & gt_one & ~pred_one
    tp_mask = valid & gt_one & pred_one
    err_mask = fp_mask | fn_mask

    inner = make_inner_mask(gt.shape, edge_margin) & valid
    edge = valid & ~inner
    gt_boundary = boundary_mask(gt_one & valid, boundary_radius)
    pred_boundary = boundary_mask(pred_one & valid, boundary_radius)
    boundary_valid = valid & (gt_boundary | pred_boundary)
    boundary_err = boundary_valid & (gt_one != pred_one)

    target_pixels = int(np.count_nonzero(valid & gt_one))
    background_pixels = int(np.count_nonzero(valid & ~gt_one))
    inner_target_pixels = int(np.count_nonzero(inner & gt_one))
    inner_background_pixels = int(np.count_nonzero(inner & ~gt_one))

    tp = int(np.count_nonzero(tp_mask))
    fp = int(np.count_nonzero(fp_mask))
    fn = int(np.count_nonzero(fn_mask))
    inner_fp = int(np.count_nonzero(fp_mask & inner))
    inner_fn = int(np.count_nonzero(fn_mask & inner))
    edge_errors = int(np.count_nonzero(err_mask & edge))
    total_errors = int(np.count_nonzero(err_mask))
    boundary_errors = int(np.count_nonzero(boundary_err))
    boundary_pixels = int(np.count_nonzero(boundary_valid))

    edge_error_share = safe_div(edge_errors, total_errors)
    boundary_error_rate = safe_div(boundary_errors, boundary_pixels)
    partial_label_suspect = edge_error_share >= 0.35

    target_iou = safe_div(tp, tp + fp + fn)
    reliable_problem_score = (
        0.45 * safe_div(inner_fn, inner_target_pixels)
        + 0.25 * safe_div(inner_fp, inner_background_pixels)
        + 0.30 * boundary_error_rate
    )
    if partial_label_suspect:
        reliable_problem_score *= 0.55

    return {
        "pixels": int(np.count_nonzero(valid)),
        "target_pixels": target_pixels,
        "background_pixels": background_pixels,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "fn_rate": safe_div(fn, target_pixels),
        "fp_rate": safe_div(fp, background_pixels),
        "inner_fp": inner_fp,
        "inner_fn": inner_fn,
        "inner_fn_rate": safe_div(inner_fn, inner_target_pixels),
        "inner_fp_rate": safe_div(inner_fp, inner_background_pixels),
        "edge_errors": edge_errors,
        "edge_error_share": edge_error_share,
        "boundary_pixels": boundary_pixels,
        "boundary_errors": boundary_errors,
        "boundary_error_rate": boundary_error_rate,
        "target_iou": target_iou,
        "partial_label_suspect": int(partial_label_suspect),
        "reliable_problem_score": reliable_problem_score,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Rank reliable hard tiles while flagging likely partial labels.")
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--pred-root", default="manual_tile_prediction_tifs_finetune_head")
    parser.add_argument("--regions", nargs="*", default=None)
    parser.add_argument("--threshold", type=float, default=0.70)
    parser.add_argument("--boundary-radius", type=int, default=2)
    parser.add_argument("--edge-margin", type=int, default=24)
    parser.add_argument("--output-dir", default="manual_tile_reliable_problem_rankings")
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

            metrics = score(gt, pred, valid, args.boundary_radius, args.edge_margin)
            rows.append({
                "region": region,
                "sample": sample,
                "stem": stem,
                "threshold": args.threshold,
                **metrics,
            })

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_csv = output_dir / "all_problem_ranking.csv"
    with all_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda item: item["reliable_problem_score"], reverse=True))

    reliable = [row for row in rows if not row["partial_label_suspect"]]
    partial = [row for row in rows if row["partial_label_suspect"]]
    reliable_csv = output_dir / "reliable_problem_tiles.csv"
    partial_csv = output_dir / "partial_label_suspect_tiles.csv"
    for path, subset, key in [
        (reliable_csv, reliable, "reliable_problem_score"),
        (partial_csv, partial, "edge_error_share"),
    ]:
        with path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(sorted(subset, key=lambda item: item[key], reverse=True))

    print(f"[INFO] Saved all ranking: {all_csv}")
    print(f"[INFO] Saved reliable problems: {reliable_csv} (n={len(reliable)})")
    print(f"[INFO] Saved partial-label suspects: {partial_csv} (n={len(partial)})")

    print("\nTop reliable problem tiles:")
    for row in sorted(reliable, key=lambda item: item["reliable_problem_score"], reverse=True)[:15]:
        print(
            f"{row['region']}/{row['sample']} score={row['reliable_problem_score']:.3f} "
            f"inner_fn={row['inner_fn']} inner_fn_rate={row['inner_fn_rate']:.3f} "
            f"boundary={row['boundary_error_rate']:.3f} IoU={row['target_iou']:.3f}"
        )

    print("\nTop partial-label suspect tiles:")
    for row in sorted(partial, key=lambda item: item["edge_error_share"], reverse=True)[:10]:
        print(
            f"{row['region']}/{row['sample']} edge_share={row['edge_error_share']:.3f} "
            f"fn={row['fn']} fp={row['fp']} boundary={row['boundary_error_rate']:.3f}"
        )


if __name__ == "__main__":
    main()

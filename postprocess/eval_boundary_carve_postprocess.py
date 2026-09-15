import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from test_manual_tiles import DEFAULT_DATA_ROOT, IGNORE_LABEL, confusion_matrix, load_multitemporal_image, metrics_from_cm
from tif_binary_postprocess import read_single_band, write_single_band


def load_split_samples(split_file, split_name):
    with Path(split_file).open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload[split_name]


def stretch_rgb_u8(image):
    data = image.reshape(6, 6, image.shape[-2], image.shape[-1])
    rgb = np.stack([data[4, 2], data[4, 1], data[4, 0]], axis=-1)
    out = np.zeros_like(rgb, dtype=np.float32)
    for channel in range(3):
        values = rgb[..., channel]
        valid = values > 0
        if np.any(valid):
            lo, hi = np.percentile(values[valid], [2, 98])
        else:
            lo, hi = 0.0, 1.0
        out[..., channel] = np.clip((values - lo) / max(hi - lo, 1e-6), 0, 1)
    return (out * 255).astype(np.uint8)


def ndvi_features(image):
    data = image.reshape(6, 6, image.shape[-2], image.shape[-1])
    red = data[:, 2]
    nir = data[:, 3]
    ndvi = (nir - red) / np.maximum(nir + red, 1e-6)
    return np.nanmean(ndvi, axis=0).astype(np.float32), np.nanmax(ndvi, axis=0).astype(np.float32)


def edge_mask_from_image(image, canny_low, canny_high, line_width):
    rgb = stretch_rgb_u8(image)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    rgb_edges = cv2.Canny(gray, canny_low, canny_high)

    ndvi_mean, _ = ndvi_features(image)
    ndvi_u8 = np.clip((ndvi_mean + 0.2) / 1.1 * 255, 0, 255).astype(np.uint8)
    ndvi_u8 = cv2.GaussianBlur(ndvi_u8, (3, 3), 0)
    ndvi_edges = cv2.Canny(ndvi_u8, canny_low, canny_high)

    edges = (rgb_edges > 0) | (ndvi_edges > 0)
    if line_width > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (line_width, line_width))
        edges = cv2.dilate(edges.astype(np.uint8), kernel, iterations=1) > 0
    return edges


def boundary_carve(image, prob, threshold, canny_low, canny_high, line_width, keep_prob, ndvi_floor):
    pred = prob >= threshold
    edges = edge_mask_from_image(image, canny_low, canny_high, line_width)
    ndvi_mean, ndvi_max = ndvi_features(image)
    low_veg_or_conf = (prob < keep_prob) | (ndvi_mean < ndvi_floor) | (ndvi_max < ndvi_floor + 0.18)
    carve = pred & edges & low_veg_or_conf
    refined = pred.copy()
    refined[carve] = False
    return refined, carve, edges


def eval_binary(pred, gt):
    out = np.full(gt.shape, IGNORE_LABEL, dtype=np.uint8)
    valid = gt != IGNORE_LABEL
    out[valid] = pred[valid].astype(np.uint8)
    cm = confusion_matrix(out, gt)
    return cm


def read_case(data_root, pred_root, item):
    region = item["region"]
    sample = item["sample"]
    stem = Path(sample).stem
    region_dir = Path(data_root) / region
    image, valid = load_multitemporal_image(region_dir, sample)
    prob, profile = read_single_band(Path(pred_root) / region / "prob" / f"{stem}_prob.tif")
    gt, _ = read_single_band(region_dir / "mask" / sample)
    prob = prob.astype(np.float32)
    if prob.max() > 1.0:
        prob = prob / 255.0
    gt = (gt > 0).astype(np.int64)
    gt[~valid] = IGNORE_LABEL
    return region, stem, image, prob, gt, profile


def overlay(rgb, mask, color, alpha):
    out = rgb.astype(np.float32) / 255.0
    out[mask] = (1 - alpha) * out[mask] + alpha * np.asarray(color, dtype=np.float32)
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)


def save_compare(path, region, stem, image, gt, pred_before, pred_after, carve):
    rgb = stretch_rgb_u8(image)
    valid = gt != IGNORE_LABEL
    gt_one = gt == 1
    before_err = np.zeros((*gt.shape, 3), dtype=np.uint8)
    after_err = np.zeros((*gt.shape, 3), dtype=np.uint8)

    for pred, err in [(pred_before, before_err), (pred_after, after_err)]:
        tp = valid & gt_one & pred
        fp = valid & ~gt_one & pred
        fn = valid & gt_one & ~pred
        err[tp] = [0, 205, 70]
        err[fp] = [40, 90, 255]
        err[fn] = [255, 40, 30]

    pred_before_vis = np.zeros((*gt.shape, 3), dtype=np.uint8)
    pred_after_vis = np.zeros((*gt.shape, 3), dtype=np.uint8)
    gt_vis = np.zeros((*gt.shape, 3), dtype=np.uint8)
    pred_before_vis[pred_before] = [0, 210, 255]
    pred_after_vis[pred_after] = [0, 210, 255]
    gt_vis[gt_one] = [255, 255, 255]
    carve_overlay = overlay(rgb, carve, [1.0, 0.85, 0.0], 0.9)

    panels = [
        ("RGB reference", rgb),
        ("GT mask white=target", gt_vis),
        ("Before prediction", pred_before_vis),
        ("After boundary carve", pred_after_vis),
        ("Before error", before_err),
        ("After error + carved yellow", overlay(after_err, carve, [1.0, 0.85, 0.0], 0.75)),
        ("Carved boundary overlay yellow", carve_overlay),
    ]
    tile_h, tile_w = gt.shape
    title_h = 24
    header_h = 34
    cols = 3
    rows = 3
    canvas = Image.new("RGB", (tile_w * cols, header_h + (tile_h + title_h) * rows), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 8), f"{region}/{stem} boundary carve comparison", fill=(0, 0, 0))
    for idx, (title, img) in enumerate(panels):
        x = (idx % cols) * tile_w
        y = header_h + (idx // cols) * (tile_h + title_h)
        draw.text((x + 6, y + 5), title, fill=(0, 0, 0))
        canvas.paste(Image.fromarray(img), (x, y + title_h))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def parse_args():
    parser = argparse.ArgumentParser(description="Image-edge boundary carving postprocess.")
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--pred-root", default="manual_tile_prediction_tifs_finetune_head")
    parser.add_argument("--split-file", default="manual_tiles_filtered_problem_removed/filtered_split.json")
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--output-dir", default="manual_tile_boundary_carve_postprocess")
    parser.add_argument("--thresholds", default="0.65")
    parser.add_argument("--canny-lows", default="35,50")
    parser.add_argument("--canny-highs", default="90,120")
    parser.add_argument("--line-widths", default="1,2")
    parser.add_argument("--keep-probs", default="0.80,0.90,0.98")
    parser.add_argument("--ndvi-floors", default="0.25,0.35")
    parser.add_argument("--save-best-tifs", action="store_true")
    parser.add_argument("--sample-region", default="阳信县4镇")
    parser.add_argument("--sample-stem", default="阳信县4镇_tile_r0006_c0006_y00768_x00768")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    samples = load_split_samples(args.split_file, args.split)
    cases = [read_case(args.data_root, args.pred_root, item) for item in samples]

    thresholds = [float(x) for x in args.thresholds.split(",") if x]
    canny_lows = [int(x) for x in args.canny_lows.split(",") if x]
    canny_highs = [int(x) for x in args.canny_highs.split(",") if x]
    line_widths = [int(x) for x in args.line_widths.split(",") if x]
    keep_probs = [float(x) for x in args.keep_probs.split(",") if x]
    ndvi_floors = [float(x) for x in args.ndvi_floors.split(",") if x]

    baseline_cm = np.zeros((2, 2), dtype=np.int64)
    for _, _, _, prob, gt, _ in cases:
        baseline_cm += eval_binary(prob >= thresholds[0], gt)
    baseline = metrics_from_cm(baseline_cm)

    rows = []
    best = None
    for threshold in thresholds:
        for canny_low in canny_lows:
            for canny_high in canny_highs:
                if canny_high <= canny_low:
                    continue
                for line_width in line_widths:
                    for keep_prob in keep_probs:
                        for ndvi_floor in ndvi_floors:
                            cm_total = np.zeros((2, 2), dtype=np.int64)
                            carved_pixels = 0
                            for _, _, image, prob, gt, _ in cases:
                                refined, carve, _ = boundary_carve(
                                    image, prob, threshold, canny_low, canny_high, line_width, keep_prob, ndvi_floor
                                )
                                carved_pixels += int(np.count_nonzero(carve))
                                cm_total += eval_binary(refined, gt)
                            metrics = metrics_from_cm(cm_total)
                            row = {
                                "threshold": threshold,
                                "canny_low": canny_low,
                                "canny_high": canny_high,
                                "line_width": line_width,
                                "keep_prob": keep_prob,
                                "ndvi_floor": ndvi_floor,
                                "carved_pixels": carved_pixels,
                                "oa": metrics["oa"],
                                "miou": metrics["miou"],
                                "target_iou": metrics["iou"][1],
                                "target_f1": metrics["f1"][1],
                                "fp": int(cm_total[0, 1]),
                                "fn": int(cm_total[1, 0]),
                                "confusion_matrix": cm_total.tolist(),
                            }
                            rows.append(row)
                            if best is None or row["miou"] > best["miou"]:
                                best = row

    csv_path = output_dir / f"{args.split}_boundary_carve_sweep.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    if args.save_best_tifs:
        for region, stem, image, prob, gt, profile in cases:
            refined, carve, _ = boundary_carve(
                image,
                prob,
                best["threshold"],
                int(best["canny_low"]),
                int(best["canny_high"]),
                int(best["line_width"]),
                best["keep_prob"],
                best["ndvi_floor"],
            )
            out = np.full(gt.shape, IGNORE_LABEL, dtype=np.uint8)
            valid = gt != IGNORE_LABEL
            out[valid] = refined[valid].astype(np.uint8)
            write_single_band(output_dir / region / f"{stem}_pred_boundary_carve.tif", out, profile)

    for region, stem, image, prob, gt, _ in cases:
        if region == args.sample_region and stem == args.sample_stem:
            refined, carve, _ = boundary_carve(
                image,
                prob,
                best["threshold"],
                int(best["canny_low"]),
                int(best["canny_high"]),
                int(best["line_width"]),
                best["keep_prob"],
                best["ndvi_floor"],
            )
            save_compare(
                output_dir / region / f"{stem}_boundary_carve_compare.png",
                region,
                stem,
                image,
                gt,
                prob >= best["threshold"],
                refined,
                carve,
            )
            break

    print("BASELINE", {
        "threshold": thresholds[0],
        "cm": baseline_cm.tolist(),
        "oa": baseline["oa"],
        "miou": baseline["miou"],
        "target_iou": baseline["iou"][1],
        "target_f1": baseline["f1"][1],
        "fp": int(baseline_cm[0, 1]),
        "fn": int(baseline_cm[1, 0]),
    })
    print("BEST", best)
    print(f"[INFO] Saved sweep: {csv_path}")


if __name__ == "__main__":
    main()

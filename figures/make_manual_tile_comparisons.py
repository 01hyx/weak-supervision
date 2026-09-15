import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from rasterio.errors import RasterioIOError
from tqdm import tqdm

from test_manual_tiles import (
    DEFAULT_DATA_ROOT,
    DEFAULT_STRIDE,
    IGNORE_LABEL,
    WINDOW_SIZE,
    build_model,
    collect_valid_sample_ids,
    confusion_matrix,
    discover_regions,
    load_checkpoint,
    load_sample,
    metrics_from_cm,
)
from threshold_sweep_manual_tiles import DEFAULT_CKPT, infer_one_image_target_prob


DEFAULT_OUTPUT_DIR = "manual_tile_comparisons"


def percentile_stretch(channel, low=2, high=98):
    valid = channel[np.isfinite(channel)]
    if valid.size == 0:
        return np.zeros_like(channel, dtype=np.uint8)
    lo, hi = np.percentile(valid, [low, high])
    if hi <= lo:
        hi = lo + 1e-6
    out = (channel - lo) / (hi - lo)
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)


def make_rgb(image_chw, time_index, rgb_bands):
    start = time_index * 6
    channels = []
    for band in rgb_bands:
        channels.append(percentile_stretch(image_chw[start + band]))
    return np.stack(channels, axis=-1)


def color_mask(mask, color):
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    out[mask == 1] = color
    out[mask == IGNORE_LABEL] = (90, 90, 90)
    return out


def make_error_map(pred, gt):
    valid = gt != IGNORE_LABEL
    out = np.zeros((*gt.shape, 3), dtype=np.uint8)
    out[~valid] = (90, 90, 90)
    out[valid & (gt == 1) & (pred == 1)] = (30, 190, 90)    # TP
    out[valid & (gt == 0) & (pred == 1)] = (230, 70, 70)    # FP
    out[valid & (gt == 1) & (pred == 0)] = (60, 120, 240)   # FN
    out[valid & (gt == 0) & (pred == 0)] = (25, 25, 25)     # TN
    return out


def resize_panel(arr, size):
    return Image.fromarray(arr).resize((size, size), Image.Resampling.NEAREST)


def draw_title(panel, title):
    header_h = 26
    out = Image.new("RGB", (panel.width, panel.height + header_h), (245, 245, 245))
    out.paste(panel, (0, header_h))
    draw = ImageDraw.Draw(out)
    font = ImageFont.load_default()
    draw.text((8, 7), title, fill=(20, 20, 20), font=font)
    return out


def make_comparison_canvas(rgb, gt, pred, prob, sample_title, panel_size):
    gt_img = color_mask(gt.astype(np.uint8), (255, 255, 255))
    pred_img = color_mask(pred.astype(np.uint8), (255, 230, 40))
    err_img = make_error_map(pred, gt)
    prob_img = np.stack([percentile_stretch(prob, 1, 99)] * 3, axis=-1)

    panels = [
        draw_title(resize_panel(rgb, panel_size), "RGB reference"),
        draw_title(resize_panel(gt_img, panel_size), "GT mask white=target"),
        draw_title(resize_panel(pred_img, panel_size), "Prediction yellow=target"),
        draw_title(resize_panel(err_img, panel_size), "Error green=TP red=FP blue=FN"),
        draw_title(resize_panel(prob_img, panel_size), "Target probability"),
    ]

    gap = 10
    title_h = 34
    width = panel_size * 2 + gap
    height = title_h + panels[0].height * 3 + gap * 2
    canvas = Image.new("RGB", (width, height), (250, 250, 250))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), sample_title, fill=(20, 20, 20), font=ImageFont.load_default())

    positions = [
        (0, title_h),
        (panel_size + gap, title_h),
        (0, title_h + panels[0].height + gap),
        (panel_size + gap, title_h + panels[0].height + gap),
        (0, title_h + (panels[0].height + gap) * 2),
    ]
    for panel, pos in zip(panels, positions):
        canvas.paste(panel, pos)
    return canvas


def parse_rgb_bands(text):
    bands = [int(x.strip()) for x in text.split(",")]
    if len(bands) != 3 or any(b < 0 or b > 5 for b in bands):
        raise ValueError("--rgb-bands must contain three 0-based band indexes within 0..5")
    return bands


def save_manifest(items, output_dir):
    path = Path(output_dir) / "comparison_manifest.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    print(f"[INFO] Saved manifest: {path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Create visual comparisons for image, mask, prediction, and errors.")
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    parser.add_argument("--regions", nargs="*", default=None)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--threshold", type=float, default=0.12)
    parser.add_argument("--samples-per-region", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE)
    parser.add_argument("--rgb-time", type=int, default=5, help="0-based time index. Default uses w6.")
    parser.add_argument("--rgb-bands", default="2,1,0", help="0-based bands inside each w folder.")
    parser.add_argument("--panel-size", type=int, default=360)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rgb_bands = parse_rgb_bands(args.rgb_bands)

    device = torch.device(args.device)
    print(f"[INFO] Checkpoint: {args.ckpt}")
    print(f"[INFO] Threshold: {args.threshold}")
    print(f"[INFO] Device: {device}")

    net = build_model(device)
    load_checkpoint(net, args.ckpt, device)
    net.eval()

    regions = discover_regions(args.data_root, args.regions)
    manifest = []
    with torch.no_grad():
        for region in regions:
            sample_ids = collect_valid_sample_ids(region)[:args.samples_per_region]
            region_out = output_dir / region.name
            region_out.mkdir(parents=True, exist_ok=True)
            for sample_name in tqdm(sample_ids, desc=region.name):
                try:
                    image, gt = load_sample(region, sample_name, ignore_nodata=True)
                    prob = infer_one_image_target_prob(
                        net, image, device, args.stride, args.batch_size, WINDOW_SIZE
                    )
                    pred = (prob >= args.threshold).astype(np.uint8)
                    cm = confusion_matrix(pred, gt)
                    metrics = metrics_from_cm(cm)
                    rgb = make_rgb(image, args.rgb_time, rgb_bands)
                    title = (
                        f"{region.name}/{sample_name}  "
                        f"t={args.threshold:.2f} OA={metrics['oa']:.2f} MIoU={metrics['miou']:.3f}"
                    )
                    canvas = make_comparison_canvas(
                        rgb, gt, pred, prob, title, panel_size=args.panel_size
                    )
                    out_path = region_out / f"{Path(sample_name).stem}_compare.png"
                    canvas.save(out_path)
                    manifest.append({
                        "region": region.name,
                        "sample": sample_name,
                        "path": str(out_path),
                        "threshold": args.threshold,
                        "oa": metrics["oa"],
                        "miou": metrics["miou"],
                        "target_iou": metrics["iou"][1],
                        "confusion_matrix": cm.tolist(),
                    })
                except (RasterioIOError, ValueError, RuntimeError) as exc:
                    print(f"[WARN] Skip sample: {sample_name} ({exc})")

    save_manifest(manifest, output_dir)
    print(f"[INFO] Saved {len(manifest)} comparison images under: {output_dir}")


if __name__ == "__main__":
    main()

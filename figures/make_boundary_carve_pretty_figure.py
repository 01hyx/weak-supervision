import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from eval_boundary_carve_postprocess import boundary_carve
from test_manual_tiles import DEFAULT_DATA_ROOT, load_multitemporal_image
from tif_binary_postprocess import read_single_band


def stretch_rgb(image):
    data = image.reshape(6, 6, image.shape[-2], image.shape[-1])
    rgb = np.stack([data[4, 2], data[4, 1], data[4, 0]], axis=-1)
    out = np.zeros_like(rgb, dtype=np.float32)
    for channel in range(3):
        values = rgb[..., channel]
        valid = values > 0
        lo, hi = np.percentile(values[valid], [2, 98]) if np.any(valid) else (0.0, 1.0)
        out[..., channel] = np.clip((values - lo) / max(hi - lo, 1e-6), 0, 1)
    return (out * 255).astype(np.uint8)


def contour_mask(mask, width=1):
    mask_u8 = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    canvas = np.zeros(mask.shape, dtype=np.uint8)
    cv2.drawContours(canvas, contours, -1, 255, width)
    return canvas > 0


def overlay_color(rgb, mask, color, alpha=0.7):
    out = rgb.astype(np.float32)
    color_arr = np.asarray(color, dtype=np.float32)
    out[mask] = (1 - alpha) * out[mask] + alpha * color_arr
    return np.clip(out, 0, 255).astype(np.uint8)


def add_label(draw, xy, text, fill=(20, 20, 20)):
    x, y = xy
    draw.rounded_rectangle((x, y, x + 214, y + 22), radius=3, fill=(255, 255, 255), outline=(220, 220, 220))
    draw.text((x + 7, y + 5), text, fill=fill)


def make_panel(rgb, gt, pred_before, pred_after, carve, mode):
    if mode == "rgb":
        return rgb
    if mode == "mask":
        out = np.zeros_like(rgb)
        out[gt] = [255, 255, 255]
        return out
    if mode == "after_mask":
        out = np.zeros_like(rgb)
        out[pred_after] = [255, 255, 255]
        return out
    if mode == "gt":
        out = rgb.copy()
        out = overlay_color(out, gt, (255, 255, 255), 0.26)
        out = overlay_color(out, contour_mask(gt, 2), (255, 225, 0), 0.98)
        return out
    if mode == "before":
        out = rgb.copy()
        out = overlay_color(out, pred_before, (255, 0, 210), 0.26)
        out = overlay_color(out, contour_mask(pred_before, 2), (255, 0, 210), 0.96)
        return out
    if mode == "after":
        out = rgb.copy()
        out = overlay_color(out, pred_after, (0, 230, 255), 0.24)
        out = overlay_color(out, contour_mask(pred_after, 2), (0, 230, 255), 0.96)
        out = overlay_color(out, carve, (255, 125, 0), 0.92)
        return out
    if mode == "compare":
        out = rgb.copy()
        gt_edge = contour_mask(gt, 2)
        before_edge = contour_mask(pred_before, 2)
        after_edge = contour_mask(pred_after, 2)
        out = overlay_color(out, before_edge, (255, 0, 210), 0.96)
        out = overlay_color(out, after_edge, (0, 230, 255), 0.96)
        out = overlay_color(out, gt_edge, (255, 225, 0), 0.98)
        out = overlay_color(out, carve, (255, 125, 0), 0.92)
        return out
    if mode == "error":
        valid = np.ones(gt.shape, dtype=bool)
        tp = valid & gt & pred_after
        fp = valid & ~gt & pred_after
        fn = valid & gt & ~pred_after
        out = rgb.copy()
        out = overlay_color(out, tp, (255, 255, 255), 0.20)
        out = overlay_color(out, fp, (0, 95, 255), 0.68)
        out = overlay_color(out, fn, (255, 35, 35), 0.72)
        return out
    raise ValueError(mode)


def parse_args():
    parser = argparse.ArgumentParser(description="Create a polished boundary-carve visualization.")
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--pred-root", default="manual_tile_prediction_tifs_finetune_head")
    parser.add_argument("--region", default="阳信县4镇")
    parser.add_argument("--stem", default="阳信县4镇_tile_r0006_c0006_y00768_x00768")
    parser.add_argument("--output-dir", default="manual_tile_boundary_carve_pretty_figures")
    parser.add_argument("--threshold", type=float, default=0.65)
    parser.add_argument("--canny-low", type=int, default=60)
    parser.add_argument("--canny-high", type=int, default=180)
    parser.add_argument("--line-width", type=int, default=1)
    parser.add_argument("--keep-prob", type=float, default=0.68)
    parser.add_argument("--ndvi-floor", type=float, default=0.15)
    return parser.parse_args()


def main():
    args = parse_args()
    sample = f"{args.stem}.tif"
    region_dir = Path(args.data_root) / args.region
    image, valid = load_multitemporal_image(region_dir, sample)
    gt, _ = read_single_band(region_dir / "mask" / sample)
    prob, _ = read_single_band(Path(args.pred_root) / args.region / "prob" / f"{args.stem}_prob.tif")
    prob = prob.astype(np.float32)
    if prob.max() > 1.0:
        prob = prob / 255.0
    gt = (gt > 0) & valid
    pred_before = (prob >= args.threshold) & valid
    pred_after, carve, _ = boundary_carve(
        image,
        prob,
        args.threshold,
        args.canny_low,
        args.canny_high,
        args.line_width,
        args.keep_prob,
        args.ndvi_floor,
    )
    pred_after &= valid
    carve &= valid

    rgb = stretch_rgb(image)
    panels = [
        ("RGB reference", make_panel(rgb, gt, pred_before, pred_after, carve, "rgb")),
        ("GT boundary", make_panel(rgb, gt, pred_before, pred_after, carve, "gt")),
        ("Before: model mask", make_panel(rgb, gt, pred_before, pred_after, carve, "before")),
        ("After: boundary carved", make_panel(rgb, gt, pred_before, pred_after, carve, "after")),
        ("GT mask", make_panel(rgb, gt, pred_before, pred_after, carve, "mask")),
        ("After mask", make_panel(rgb, gt, pred_before, pred_after, carve, "after_mask")),
        ("Boundary overlay", make_panel(rgb, gt, pred_before, pred_after, carve, "compare")),
        ("After errors", make_panel(rgb, gt, pred_before, pred_after, carve, "error")),
    ]

    tile_h, tile_w = gt.shape
    gutter = 14
    label_h = 28
    header_h = 52
    cols = 3
    rows = 3
    canvas_w = cols * tile_w + (cols + 1) * gutter
    canvas_h = header_h + rows * (tile_h + label_h) + (rows + 1) * gutter + 34
    canvas = Image.new("RGB", (canvas_w, canvas_h), (246, 247, 248))
    draw = ImageDraw.Draw(canvas)
    draw.text((gutter, 14), f"{args.region} / {args.stem}", fill=(20, 20, 20))
    draw.text((gutter, 32), "Yellow=GT  Magenta=before  Cyan=after  Orange=carved field lines", fill=(75, 75, 75))

    for idx, (title, img) in enumerate(panels):
        col = idx % cols
        row = idx // cols
        x = gutter + col * (tile_w + gutter)
        y = header_h + gutter + row * (tile_h + label_h + gutter)
        card = Image.new("RGB", (tile_w, tile_h + label_h), (255, 255, 255))
        card_draw = ImageDraw.Draw(card)
        card_draw.text((8, 8), title, fill=(20, 20, 20))
        card.paste(Image.fromarray(img), (0, label_h))
        canvas.paste(card, (x, y))

    out_dir = Path(args.output_dir) / args.region
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.stem}_pretty_boundary_carve.png"
    canvas.save(out_path)
    print(f"[INFO] Saved: {out_path}")


if __name__ == "__main__":
    main()

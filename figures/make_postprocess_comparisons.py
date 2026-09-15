import argparse
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image, ImageDraw, ImageFont

from make_manual_tile_comparisons import (
    draw_title,
    make_error_map,
    make_rgb,
    resize_panel,
)
from test_manual_tiles import DEFAULT_DATA_ROOT, IGNORE_LABEL, load_sample, metrics_from_cm, confusion_matrix


def read_tif(path):
    with rasterio.open(path) as src:
        return src.read(1)


def color_prediction(mask, color):
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    out[mask == 1] = color
    out[mask == IGNORE_LABEL] = (90, 90, 90)
    return out


def make_canvas(rgb, gt, pred, sessrs, post, title, panel_size):
    gt_img = color_prediction(gt.astype(np.uint8), (255, 255, 255))
    pred_img = color_prediction(pred.astype(np.uint8), (255, 230, 40))
    sessrs_img = color_prediction(sessrs.astype(np.uint8), (180, 80, 255))
    post_img = color_prediction(post.astype(np.uint8), (40, 210, 255))
    err_img = make_error_map(pred, gt)
    sessrs_err_img = make_error_map(sessrs, gt)
    post_err_img = make_error_map(post, gt)

    panels = [
        draw_title(resize_panel(rgb, panel_size), "RGB reference"),
        draw_title(resize_panel(gt_img, panel_size), "GT mask white=target"),
        draw_title(resize_panel(pred_img, panel_size), "Prediction yellow=target"),
        draw_title(resize_panel(sessrs_img, panel_size), "SESSRS result purple=target"),
        draw_title(resize_panel(post_img, panel_size), "Postprocess cyan=target"),
        draw_title(resize_panel(err_img, panel_size), "Error before green=TP red=FP blue=FN"),
        draw_title(resize_panel(sessrs_err_img, panel_size), "Error SESSRS green=TP red=FP blue=FN"),
        draw_title(resize_panel(post_err_img, panel_size), "Error after green=TP red=FP blue=FN"),
    ]

    gap = 10
    title_h = 34
    panel_h = panels[0].height
    width = panel_size * 2 + gap
    height = title_h + panel_h * 4 + gap * 3
    canvas = Image.new("RGB", (width, height), (250, 250, 250))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), title, fill=(20, 20, 20), font=ImageFont.load_default())

    positions = [
        (0, title_h),
        (panel_size + gap, title_h),
        (0, title_h + panel_h + gap),
        (panel_size + gap, title_h + panel_h + gap),
        (0, title_h + (panel_h + gap) * 2),
        (panel_size + gap, title_h + (panel_h + gap) * 2),
        (0, title_h + (panel_h + gap) * 3),
        (panel_size + gap, title_h + (panel_h + gap) * 3),
    ]
    for panel, pos in zip(panels, positions):
        canvas.paste(panel, pos)
    return canvas


def parse_rgb_bands(text):
    bands = [int(x.strip()) for x in text.split(",")]
    if len(bands) != 3 or any(b < 0 or b > 5 for b in bands):
        raise ValueError("--rgb-bands must contain three 0-based band indexes within 0..5")
    return bands


def parse_args():
    parser = argparse.ArgumentParser(description="Create before/after postprocess comparison panels.")
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--region", required=True)
    parser.add_argument("--stems", nargs="+", required=True)
    parser.add_argument("--pred-root", required=True)
    parser.add_argument("--post-root", required=True)
    parser.add_argument("--output-dir", default="manual_tile_postprocess_comparisons")
    parser.add_argument("--rgb-time", type=int, default=5)
    parser.add_argument("--rgb-bands", default="2,1,0")
    parser.add_argument("--panel-size", type=int, default=360)
    return parser.parse_args()


def main():
    args = parse_args()
    rgb_bands = parse_rgb_bands(args.rgb_bands)
    data_region = Path(args.data_root) / args.region
    pred_region = Path(args.pred_root) / args.region
    post_region = Path(args.post_root) / args.region
    out_region = Path(args.output_dir) / args.region
    out_region.mkdir(parents=True, exist_ok=True)

    for stem in args.stems:
        sample_name = f"{stem}.tif"
        image, gt = load_sample(data_region, sample_name, ignore_nodata=True)
        pred = read_tif(pred_region / "pred" / f"{stem}_pred.tif").astype(np.uint8)
        sessrs = read_tif(post_region / f"{stem}_pred_sessrs.tif").astype(np.uint8)
        post = read_tif(post_region / f"{stem}_pred_post.tif").astype(np.uint8)
        rgb = make_rgb(image, args.rgb_time, rgb_bands)

        before = metrics_from_cm(confusion_matrix(pred, gt))
        middle = metrics_from_cm(confusion_matrix(sessrs, gt))
        after = metrics_from_cm(confusion_matrix(post, gt))
        title = (
            f"{args.region}/{sample_name}  "
            f"MIoU {before['miou']:.3f}->{middle['miou']:.3f}->{after['miou']:.3f}  "
            f"targetIoU {before['iou'][1]:.3f}->{middle['iou'][1]:.3f}->{after['iou'][1]:.3f}"
        )
        canvas = make_canvas(rgb, gt, pred, sessrs, post, title, args.panel_size)
        out_path = out_region / f"{stem}_post_compare.png"
        canvas.save(out_path)
        print(f"[INFO] Saved: {out_path}")


if __name__ == "__main__":
    main()

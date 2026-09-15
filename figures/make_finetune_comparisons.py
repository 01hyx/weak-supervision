import argparse
from pathlib import Path

import numpy as np
import rasterio
import torch
from PIL import Image, ImageDraw, ImageFont

from make_manual_tile_comparisons import (
    draw_title,
    make_error_map,
    make_rgb,
    percentile_stretch,
    resize_panel,
)
from test_manual_tiles import DEFAULT_DATA_ROOT, WINDOW_SIZE, build_model, confusion_matrix, load_sample, metrics_from_cm
from threshold_sweep_manual_tiles import DEFAULT_CKPT, infer_one_image_target_prob


def read_prob_tif(path):
    with rasterio.open(path) as src:
        return src.read(1).astype(np.float32)


def color_prediction(mask, color):
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    out[mask == 1] = color
    out[mask == 255] = (90, 90, 90)
    return out


def prob_to_rgb(prob):
    gray = percentile_stretch(prob, 1, 99)
    return np.stack([gray, gray, gray], axis=-1)


def make_canvas(rgb, gt, orig_pred, ft_pred, orig_prob, ft_prob, title, panel_size):
    gt_img = color_prediction(gt.astype(np.uint8), (255, 255, 255))
    orig_img = color_prediction(orig_pred.astype(np.uint8), (255, 230, 40))
    ft_img = color_prediction(ft_pred.astype(np.uint8), (40, 210, 255))
    orig_err = make_error_map(orig_pred, gt)
    ft_err = make_error_map(ft_pred, gt)

    panels = [
        draw_title(resize_panel(rgb, panel_size), "RGB reference"),
        draw_title(resize_panel(gt_img, panel_size), "GT mask white=target"),
        draw_title(resize_panel(orig_img, panel_size), "Original prediction yellow=target"),
        draw_title(resize_panel(ft_img, panel_size), "Fine-tuned prediction cyan=target"),
        draw_title(resize_panel(orig_err, panel_size), "Original error green=TP red=FP blue=FN"),
        draw_title(resize_panel(ft_err, panel_size), "Fine-tuned error green=TP red=FP blue=FN"),
        draw_title(resize_panel(prob_to_rgb(orig_prob), panel_size), "Original target probability"),
        draw_title(resize_panel(prob_to_rgb(ft_prob), panel_size), "Fine-tuned target probability"),
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
    parser = argparse.ArgumentParser(description="Compare original RS3Mamba and fine-tuned predictions.")
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--region", required=True)
    parser.add_argument("--stems", nargs="+", required=True)
    parser.add_argument("--original-prob-root", default="manual_tile_prediction_tifs_all")
    parser.add_argument("--original-ckpt", default=str(DEFAULT_CKPT))
    parser.add_argument("--finetuned-ckpt", default="manual_tiles_finetune_head_trial3/best.pth")
    parser.add_argument("--output-dir", default="manual_tile_finetune_comparisons")
    parser.add_argument("--original-threshold", type=float, default=0.14)
    parser.add_argument("--finetuned-threshold", type=float, default=0.70)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--rgb-time", type=int, default=5)
    parser.add_argument("--rgb-bands", default="2,1,0")
    parser.add_argument("--panel-size", type=int, default=360)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    rgb_bands = parse_rgb_bands(args.rgb_bands)
    device = torch.device(args.device)
    data_region = Path(args.data_root) / args.region
    out_region = Path(args.output_dir) / args.region
    out_region.mkdir(parents=True, exist_ok=True)

    net = build_model(device)
    state = torch.load(args.finetuned_ckpt, map_location=device)
    net.load_state_dict(state, strict=False)
    net.eval()

    print(f"[INFO] Fine-tuned ckpt: {args.finetuned_ckpt}")
    print(f"[INFO] Original threshold: {args.original_threshold}")
    print(f"[INFO] Fine-tuned threshold: {args.finetuned_threshold}")

    with torch.no_grad():
        for stem in args.stems:
            sample_name = f"{stem}.tif"
            image, gt = load_sample(data_region, sample_name, ignore_nodata=True)
            rgb = make_rgb(image, args.rgb_time, rgb_bands)

            orig_prob_path = Path(args.original_prob_root) / args.region / "prob" / f"{stem}_prob.tif"
            if orig_prob_path.exists():
                orig_prob = read_prob_tif(orig_prob_path)
            else:
                raise FileNotFoundError(f"Original probability tif not found: {orig_prob_path}")
            orig_pred = (orig_prob >= args.original_threshold).astype(np.uint8)

            ft_prob = infer_one_image_target_prob(net, image, device, args.stride, args.batch_size, WINDOW_SIZE)
            ft_pred = (ft_prob >= args.finetuned_threshold).astype(np.uint8)

            orig_metrics = metrics_from_cm(confusion_matrix(orig_pred, gt))
            ft_metrics = metrics_from_cm(confusion_matrix(ft_pred, gt))
            title = (
                f"{args.region}/{sample_name}  "
                f"MIoU {orig_metrics['miou']:.3f}->{ft_metrics['miou']:.3f}  "
                f"targetIoU {orig_metrics['iou'][1]:.3f}->{ft_metrics['iou'][1]:.3f}"
            )
            canvas = make_canvas(rgb, gt, orig_pred, ft_pred, orig_prob, ft_prob, title, args.panel_size)
            out_path = out_region / f"{stem}_finetune_compare.png"
            canvas.save(out_path)
            print(f"[INFO] Saved: {out_path}")


if __name__ == "__main__":
    main()

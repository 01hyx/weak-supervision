"""输出单个样本的六期 Haar 高频边界与融合边界图。"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from eval_boundary_carve_postprocess import stretch_rgb_u8
from eval_wavelet_result_postprocess import haar_detail, normalize_robust, spatial_frequency_score
from test_manual_tiles import load_sample


def color_score(score):
    score_u8 = np.clip(score * 255, 0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.applyColorMap(score_u8, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)


def binary_edge(score, threshold):
    out = np.zeros((*score.shape, 3), dtype=np.uint8)
    out[score >= threshold] = 255
    return out


def overlay(rgb, score, threshold):
    out = rgb.copy()
    out[score >= threshold] = (0, 220, 255)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-key", default="r0008_c0010_y01024_x01280")
    parser.add_argument("--split-file", default="manual_tiles_filtered_problem_removed/filtered_trainval_split.json")
    parser.add_argument("--data-root", default=r"D:\s2_output\manual_tiles_maize30_stride128")
    parser.add_argument("--output-dir", default="wavelet_boundary_debug")
    parser.add_argument("--threshold", type=float, default=0.30)
    parser.add_argument("--panel-size", type=int, default=768)
    args = parser.parse_args()

    payload = json.loads(Path(args.split_file).read_text(encoding="utf-8"))
    items = payload["train"] + payload["val"]
    item = next(item for item in items if args.sample_key in item["sample"])
    image, _ = load_sample(Path(args.data_root) / item["region"], item["sample"], ignore_nodata=True)
    data = image.reshape(6, 6, image.shape[-2], image.shape[-1])
    rgb = stretch_rgb_u8(image)

    rows = []
    for index in range(6):
        red = data[index, 2]
        nir = data[index, 3]
        ndvi = (nir - red) / np.maximum(nir + red, 1e-6)
        brightness = np.mean(data[index, :4], axis=0)
        rgb_score = normalize_robust(haar_detail(brightness))
        ndvi_score = normalize_robust(haar_detail(ndvi))
        combined = 0.5 * rgb_score + 0.5 * ndvi_score
        rows.append([color_score(rgb_score), color_score(ndvi_score), binary_edge(combined, args.threshold), overlay(rgb, combined, args.threshold)])

    fused = spatial_frequency_score(image)
    rows.append([color_score(fused), binary_edge(fused, args.threshold), overlay(rgb, fused, args.threshold), rgb])

    size, gap = args.panel_size, 12
    canvas = Image.new("RGB", (size * 4 + gap * 3, size * 7 + gap * 6), "white")
    for row_index, panels in enumerate(rows):
        for col_index, panel in enumerate(panels):
            resized = Image.fromarray(panel).resize((size, size), Image.Resampling.NEAREST)
            canvas.paste(resized, (col_index * (size + gap), row_index * (size + gap)))

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"{Path(item['sample']).stem}_wavelet_boundaries.png"
    canvas.save(path, dpi=(300, 300), compress_level=4)
    print(path.resolve())


if __name__ == "__main__":
    main()

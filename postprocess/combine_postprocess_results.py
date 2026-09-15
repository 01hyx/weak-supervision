"""合并外边界修正结果与内部低植被走廊切分结果。"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from eval_boundary_carve_postprocess import stretch_rgb_u8
from eval_internal_frequency_boundaries import eval_pred
from test_manual_tiles import DEFAULT_DATA_ROOT, IGNORE_LABEL, load_sample, metrics_from_cm
from tif_binary_postprocess import read_single_band, write_single_band


def save_figure(path, image, gt, baseline, outer, combined):
    valid = gt != IGNORE_LABEL
    panels = [stretch_rgb_u8(image)]
    for mask in (gt == 1, baseline, outer, combined):
        panel = np.zeros((*gt.shape, 3), dtype=np.uint8)
        panel[mask & valid] = 255
        panel[~valid] = 70
        panels.append(panel)
    size, gap = 768, 14
    canvas = Image.new("RGB", (size * 5 + gap * 4, size), "white")
    for index, panel in enumerate(panels):
        canvas.paste(Image.fromarray(panel).resize((size, size), Image.Resampling.NEAREST), (index * (size + gap), 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, dpi=(300, 300), compress_level=4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-file", default="manual_tiles_filtered_problem_removed/filtered_trainval_split.json")
    parser.add_argument("--split-part", choices=["train", "val"], default="val")
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--prob-root", default="frequency_best_prediction_tifs_val")
    parser.add_argument("--outer-root", default="frequency_wavelet_result_postprocess_refined")
    parser.add_argument("--corridor-root", default="temporal_low_vegetation_corridors_fast")
    parser.add_argument("--output-dir", default="combined_outer_and_internal_postprocess")
    args = parser.parse_args()

    items = json.loads(Path(args.split_file).read_text(encoding="utf-8"))[args.split_part]
    cms = {name: np.zeros((2, 2), dtype=np.int64) for name in ("baseline", "outer", "corridor", "combined")}
    output = Path(args.output_dir)
    for item in items:
        region, sample = item["region"], item["sample"]
        stem = Path(sample).stem
        image, gt = load_sample(Path(args.data_root) / region, sample, ignore_nodata=True)
        prob, _ = read_single_band(Path(args.prob_root) / region / "prob" / f"{stem}_prob.tif")
        outer_raw, profile = read_single_band(Path(args.outer_root) / region / "pred" / f"{stem}_pred.tif")
        corridor_raw, _ = read_single_band(Path(args.corridor_root) / region / "pred" / f"{stem}_pred.tif")
        baseline = prob >= 0.60
        outer = outer_raw == 1
        corridor = corridor_raw == 1
        # 内部走廊后处理只负责删线，因此以高精度外边界结果为主体取交集。
        combined = outer & corridor
        for name, pred in (("baseline", baseline), ("outer", outer), ("corridor", corridor), ("combined", combined)):
            cms[name] += eval_pred(pred, gt)
        valid = gt != IGNORE_LABEL
        out = np.full(gt.shape, IGNORE_LABEL, dtype=np.uint8)
        out[valid] = combined[valid].astype(np.uint8)
        write_single_band(output / region / "pred" / f"{stem}_pred.tif", out, profile)
        save_figure(output / region / "figures" / f"{stem}_combined_compare.png", image, gt, baseline, outer, combined)

    summary = {name: metrics_from_cm(cm) for name, cm in cms.items()}
    (output / f"{args.split_part}_combined_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

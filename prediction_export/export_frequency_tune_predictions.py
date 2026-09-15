"""导出频域增强微调模型的概率图和二值 GeoTIFF。"""

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
import torch
from tqdm import tqdm

from evaluate_adaptation_experiments import build_experiment_model
from test_manual_tiles import (
    DEFAULT_DATA_ROOT,
    IGNORE_LABEL,
    WINDOW_SIZE,
    collect_valid_sample_ids,
    discover_regions,
    load_sample,
)
from threshold_sweep_manual_tiles import infer_one_image_target_prob


def write_tif(path, array, profile, dtype, nodata=None):
    profile = profile.copy()
    profile.update(count=1, dtype=dtype, compress="lzw")
    if nodata is not None:
        profile.update(nodata=nodata)
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array.astype(dtype), 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment_frequency_wavelet.json")
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--regions", nargs="*", default=None)
    parser.add_argument("--split-file", default=None)
    parser.add_argument("--split-part", choices=["train", "val", "test"], default="val")
    parser.add_argument("--output-dir", default="frequency_tune_prediction_tifs")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", action="store_true", help="Skip samples whose probability and prediction TIFs already exist.")
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    device = torch.device(args.device)
    model = build_experiment_model(config, device)
    threshold = float(config["threshold"])
    output_dir = Path(args.output_dir)
    selected = None
    if args.split_file:
        payload = json.loads(Path(args.split_file).read_text(encoding="utf-8"))
        selected = {}
        for item in payload[args.split_part]:
            selected.setdefault(item["region"], set()).add(item["sample"])

    with torch.no_grad():
        for region in discover_regions(args.data_root, args.regions):
            samples = collect_valid_sample_ids(region)
            if selected is not None:
                samples = [sample for sample in samples if sample in selected.get(region.name, set())]
            for sample in tqdm(samples, desc=region.name):
                stem = Path(sample).stem
                region_out = output_dir / region.name
                prob_path = region_out / "prob" / f"{stem}_prob.tif"
                pred_path = region_out / "pred" / f"{stem}_pred.tif"
                if args.resume and prob_path.exists() and pred_path.exists():
                    continue
                image, gt = load_sample(region, sample, ignore_nodata=True)
                prob = infer_one_image_target_prob(
                    model, image, device, args.stride, args.batch_size, WINDOW_SIZE
                ).astype(np.float32)
                pred = (prob >= threshold).astype(np.uint8)
                pred[gt == IGNORE_LABEL] = IGNORE_LABEL
                with rasterio.open(region / "mask" / sample) as src:
                    profile = src.profile.copy()
                write_tif(prob_path, prob, profile, "float32")
                write_tif(
                    pred_path,
                    pred,
                    profile,
                    "uint8",
                    IGNORE_LABEL,
                )
    print(f"[INFO] 导出完成: {output_dir.resolve()}")


if __name__ == "__main__":
    main()

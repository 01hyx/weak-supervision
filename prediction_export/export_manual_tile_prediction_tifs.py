import argparse
from pathlib import Path

import numpy as np
import rasterio
import torch
from rasterio.errors import RasterioIOError
from tqdm import tqdm

from test_manual_tiles import (
    DEFAULT_DATA_ROOT,
    DEFAULT_STRIDE,
    IGNORE_LABEL,
    WINDOW_SIZE,
    build_model,
    collect_valid_sample_ids,
    discover_regions,
    load_checkpoint,
    load_sample,
)
from threshold_sweep_manual_tiles import DEFAULT_CKPT, infer_one_image_target_prob


DEFAULT_OUTPUT_DIR = "manual_tile_prediction_tifs"


def read_profile(path):
    with rasterio.open(path) as src:
        return src.profile.copy()


def write_tif(path, arr, profile, dtype, nodata=None):
    out_profile = profile.copy()
    out_profile.update(count=1, dtype=dtype, compress="lzw")
    if nodata is not None:
        out_profile.update(nodata=nodata)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **out_profile) as dst:
        dst.write(arr.astype(dtype), 1)


def parse_args():
    parser = argparse.ArgumentParser(description="Export RS3Mamba binary predictions and probabilities as GeoTIFF.")
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    parser.add_argument("--regions", nargs="*", default=None)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--threshold", type=float, default=0.12)
    parser.add_argument("--samples-per-region", type=int, default=0, help="0 means all samples.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    device = torch.device(args.device)

    print(f"[INFO] Checkpoint: {args.ckpt}")
    print(f"[INFO] Threshold: {args.threshold}")
    print(f"[INFO] Device: {device}")

    net = build_model(device)
    load_checkpoint(net, args.ckpt, device)
    net.eval()

    regions = discover_regions(args.data_root, args.regions)
    exported = 0
    with torch.no_grad():
        for region in regions:
            sample_ids = collect_valid_sample_ids(region)
            if args.samples_per_region > 0:
                sample_ids = sample_ids[:args.samples_per_region]

            for sample_name in tqdm(sample_ids, desc=region.name):
                try:
                    image, gt = load_sample(region, sample_name, ignore_nodata=True)
                    prob = infer_one_image_target_prob(
                        net, image, device, args.stride, args.batch_size, WINDOW_SIZE
                    ).astype(np.float32)
                    pred = (prob >= args.threshold).astype(np.uint8)
                    pred[gt == IGNORE_LABEL] = IGNORE_LABEL

                    profile = read_profile(region / "mask" / sample_name)
                    stem = Path(sample_name).stem
                    region_out = output_dir / region.name
                    write_tif(region_out / "prob" / f"{stem}_prob.tif", prob, profile, "float32", nodata=None)
                    write_tif(region_out / "pred" / f"{stem}_pred.tif", pred, profile, "uint8", nodata=IGNORE_LABEL)
                    exported += 1
                except (RasterioIOError, ValueError, RuntimeError) as exc:
                    print(f"[WARN] Skip sample: {sample_name} ({exc})")

    print(f"[INFO] Exported {exported} samples to: {output_dir}")


if __name__ == "__main__":
    main()

"""Export semantic, boundary, center and offset maps for parcel-instance decoding."""

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
import torch
from tqdm import tqdm

from model.ParcelInstanceRS3Mamba import ParcelInstanceRS3Mamba
from test_manual_tiles import BANDS_PER_STEP, IN_CHANNELS, N_CLASSES, TIME_STEPS, compute_time_quality, load_sample
from tif_binary_postprocess import write_single_band


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="qingdao_parcel_instance_heads/best.pth")
    parser.add_argument("--data-root", default=r"D:\s2_output\qingdao_selected150_2025_sentinel")
    parser.add_argument("--split-file", default="configs/qingdao_2025_spatial_split.json")
    parser.add_argument("--region", default="青岛夏玉米0925")
    parser.add_argument("--output-dir", default="qingdao_parcel_instance_outputs")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = ParcelInstanceRS3Mamba(
        num_classes=N_CLASSES,
        in_channels=IN_CHANNELS,
        pretrained=False,
        use_phenology_fusion=False,
        time_steps=TIME_STEPS,
        bands_per_step=BANDS_PER_STEP,
        phenology_prior_mode="data",
        use_frequency_enhance=True,
        frequency_module="wavelet",
        frequency_channels=64,
    ).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()

    split = json.loads(Path(args.split_file).read_text(encoding="utf-8"))
    items = [
        {**item, "split": part}
        for part in ("train", "val", "test")
        for item in split[part]
        if item["region"] == args.region
    ]
    output = Path(args.output_dir) / args.region
    with torch.no_grad():
        for item in tqdm(items, desc="Parcel instance outputs"):
            sample = item["sample"]
            stem = Path(sample).stem
            image, _ = load_sample(Path(args.data_root) / args.region, sample, ignore_nodata=True)
            positions, quality, pad_mask = compute_time_quality(image)
            outputs = model(
                torch.from_numpy(image[None].astype(np.float32)).to(device),
                batch_positions=torch.from_numpy(positions[None]).to(device),
                quality_score=torch.from_numpy(quality[None]).to(device),
                pad_mask=torch.from_numpy(pad_mask[None]).to(device),
            )
            arrays = {
                "semantic": torch.softmax(outputs["semantic"], dim=1)[0, 1].cpu().numpy(),
                "boundary": torch.sigmoid(outputs["boundary"])[0, 0].cpu().numpy(),
                "center": torch.sigmoid(outputs["center"])[0, 0].cpu().numpy(),
                "offset_x": outputs["offset"][0, 0].cpu().numpy(),
                "offset_y": outputs["offset"][0, 1].cpu().numpy(),
            }
            with rasterio.open(Path(args.data_root) / args.region / "mask" / sample) as src:
                profile = src.profile.copy()
            for name, array in arrays.items():
                write_single_band(
                    output / name / f"{stem}_{name}.tif",
                    array.astype(np.float32),
                    profile,
                    dtype="float32",
                    nodata=-9999.0,
                )
    print(f"[INFO] Exported {len(items)} samples: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()

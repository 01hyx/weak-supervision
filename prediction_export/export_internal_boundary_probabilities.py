"""Export internal parcel-separator probabilities from the auxiliary head."""

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
import torch
from tqdm import tqdm

from evaluate_adaptation_experiments import build_experiment_model
from test_manual_tiles import compute_time_quality, load_sample
from tif_binary_postprocess import write_single_band
from train_internal_boundary_head import InternalBoundaryHead


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--head-checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split-file", required=True)
    parser.add_argument("--split-parts", nargs="+", default=["train", "val"])
    parser.add_argument("--regions", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.head_checkpoint, map_location=device)
    model = build_experiment_model(checkpoint["model_config"], device)
    head = InternalBoundaryHead(64).to(device)
    head.load_state_dict(checkpoint["state_dict"])
    head.eval()
    cache = {}
    hook = model.decoder.p1.register_forward_hook(
        lambda module, inputs, output: cache.__setitem__("feature", output.detach())
    )

    split = json.loads(Path(args.split_file).read_text(encoding="utf-8"))
    items = [
        item
        for part in args.split_parts
        for item in split[part]
        if item["region"] in args.regions
    ]
    with torch.no_grad():
        for item in tqdm(items, desc="Internal boundary probabilities"):
            region, sample = item["region"], item["sample"]
            image, _ = load_sample(Path(args.data_root) / region, sample, ignore_nodata=True)
            positions, quality, pad_mask = compute_time_quality(image)
            tensor = torch.from_numpy(image[None].astype(np.float32)).to(device)
            model(
                tensor,
                batch_positions=torch.from_numpy(positions[None]).to(device),
                quality_score=torch.from_numpy(quality[None]).to(device),
                pad_mask=torch.from_numpy(pad_mask[None]).to(device),
            )
            probability = torch.sigmoid(
                head(cache["feature"], image.shape[-2:])
            )[0, 0].cpu().numpy().astype(np.float32)
            with rasterio.open(Path(args.data_root) / region / "mask" / sample) as src:
                profile = src.profile.copy()
            stem = Path(sample).stem
            write_single_band(
                Path(args.output_dir) / region / "prob" / f"{stem}_boundary_prob.tif",
                probability,
                profile,
            )
    hook.remove()
    print(f"[INFO] Exported {len(items)} boundary probability maps: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()

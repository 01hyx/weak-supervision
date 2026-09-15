"""Train parcel boundary, center and offset heads on Qingdao instance labels."""

import argparse
import csv
import json
import random
from pathlib import Path

import cv2
import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from finetune_manual_tiles import load_split_file, seed_everything
from model.FreqTuneRS3Mamba import load_base_rs3mamba_weights
from model.ParcelInstanceRS3Mamba import ParcelInstanceRS3Mamba
from test_manual_tiles import (
    BANDS_PER_STEP,
    IN_CHANNELS,
    N_CLASSES,
    TIME_STEPS,
    compute_time_quality,
    load_sample,
)


class ParcelInstanceDataset(Dataset):
    def __init__(self, samples, augmentation, minimum_area=20):
        self.samples = list(samples)
        self.augmentation = augmentation
        self.minimum_area = minimum_area

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        region_dir, sample = self.samples[index]
        image, semantic = load_sample(region_dir, sample, ignore_nodata=True)
        with rasterio.open(region_dir / "instance_mask" / sample) as src:
            instances = src.read(1).astype(np.int64)
        if self.augmentation:
            if random.random() < 0.5:
                image, semantic, instances = image[:, ::-1, :], semantic[::-1, :], instances[::-1, :]
            if random.random() < 0.5:
                image, semantic, instances = image[:, :, ::-1], semantic[:, ::-1], instances[:, ::-1]
        image = np.ascontiguousarray(image)
        semantic = np.ascontiguousarray(semantic)
        instances = np.ascontiguousarray(instances)
        boundary, center, offset, instance_valid = make_instance_targets(instances, self.minimum_area)
        positions, quality, pad_mask = compute_time_quality(image)
        return {
            "image": torch.from_numpy(image.astype(np.float32)),
            "semantic": torch.from_numpy(semantic.astype(np.int64)),
            "boundary": torch.from_numpy(boundary[None].astype(np.float32)),
            "center": torch.from_numpy(center[None].astype(np.float32)),
            "offset": torch.from_numpy(offset.astype(np.float32)),
            "instance_valid": torch.from_numpy(instance_valid[None]),
            "positions": torch.from_numpy(positions.astype(np.float32)),
            "quality": torch.from_numpy(quality.astype(np.float32)),
            "pad_mask": torch.from_numpy(pad_mask.astype(np.bool_)),
        }


def make_instance_targets(instances, minimum_area):
    height, width = instances.shape
    boundary = np.zeros((height, width), dtype=np.float32)
    center = np.zeros((height, width), dtype=np.float32)
    offset = np.zeros((2, height, width), dtype=np.float32)
    valid = np.zeros((height, width), dtype=bool)
    yy, xx = np.indices((height, width))
    gaussian_radius = 3
    for instance_id in np.unique(instances):
        if instance_id == 0:
            continue
        mask = instances == instance_id
        ys, xs = np.nonzero(mask)
        if len(xs) < minimum_area:
            continue
        if np.any(ys == 0) or np.any(xs == 0) or np.any(ys == height - 1) or np.any(xs == width - 1):
            continue
        center_x, center_y = float(xs.mean()), float(ys.mean())
        valid[mask] = True
        offset[0, mask] = (center_x - xx[mask]) / width
        offset[1, mask] = (center_y - yy[mask]) / height
        distance = (xx - center_x) ** 2 + (yy - center_y) ** 2
        center = np.maximum(center, np.exp(-distance / (2 * gaussian_radius**2)).astype(np.float32))
        gradient = cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8)) != cv2.erode(
            mask.astype(np.uint8), np.ones((3, 3), np.uint8)
        )
        boundary[gradient] = 1.0
    return boundary, center, offset, valid


def boundary_loss(logits, target, positive_weight=4.0):
    pos_weight = torch.tensor([positive_weight], device=logits.device)
    bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)
    probability = torch.sigmoid(logits)
    intersection = torch.sum(probability * target)
    dice = 1.0 - (2 * intersection + 1) / (torch.sum(probability) + torch.sum(target) + 1)
    return bce + dice


def train_or_validate(model, loader, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    # 冻结主干中的 BN 统计量，避免少量实例样本改变原语义模型分布。
    if training:
        for module in model.modules():
            if isinstance(module, torch.nn.BatchNorm2d) and not any(
                parameter.requires_grad for parameter in module.parameters()
            ):
                module.eval()
    totals = {"loss": 0.0, "boundary": 0.0, "center": 0.0, "offset": 0.0}
    batches = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in tqdm(loader, desc="instance train" if training else "instance val", leave=False):
            image = batch["image"].to(device)
            outputs = model(
                image,
                batch_positions=batch["positions"].to(device),
                quality_score=batch["quality"].to(device),
                pad_mask=batch["pad_mask"].to(device),
            )
            target_boundary = batch["boundary"].to(device)
            target_center = batch["center"].to(device)
            target_offset = batch["offset"].to(device)
            instance_valid = batch["instance_valid"].to(device)
            loss_boundary = boundary_loss(outputs["boundary"], target_boundary)
            center_probability = torch.sigmoid(outputs["center"])
            center_weight = 1.0 + 9.0 * target_center
            loss_center = torch.mean(center_weight * (center_probability - target_center) ** 2)
            valid_offset = instance_valid.expand_as(target_offset)
            if torch.any(valid_offset):
                loss_offset = F.smooth_l1_loss(outputs["offset"][valid_offset], target_offset[valid_offset])
            else:
                loss_offset = outputs["offset"].sum() * 0.0
            loss = 0.30 * loss_boundary + 0.20 * loss_center + 0.10 * loss_offset
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            totals["loss"] += float(loss.detach())
            totals["boundary"] += float(loss_boundary.detach())
            totals["center"] += float(loss_center.detach())
            totals["offset"] += float(loss_offset.detach())
            batches += 1
    return {key: value / max(batches, 1) for key, value in totals.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=r"D:\s2_output\qingdao_selected150_2025_sentinel")
    parser.add_argument("--split-file", default="configs/qingdao_2025_spatial_split.json")
    parser.add_argument("--pretrained", default="qingdao_boundary_refine/best.pth")
    parser.add_argument("--output-dir", default="qingdao_parcel_instance_heads")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--minimum-instance-area", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    seed_everything(42)
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
    load_base_rs3mamba_weights(model, args.pretrained, map_location=device)
    for name, parameter in model.named_parameters():
        parameter.requires_grad = name.startswith(
            ("instance_refine.", "parcel_boundary_head.", "parcel_center_head.", "parcel_offset_head.")
        )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.lr,
        weight_decay=1e-4,
    )
    train_samples, val_samples = load_split_file(args.split_file, args.data_root)
    train_loader = DataLoader(
        ParcelInstanceDataset(
            train_samples,
            augmentation=True,
            minimum_area=args.minimum_instance_area,
        ),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        ParcelInstanceDataset(
            val_samples,
            augmentation=False,
            minimum_area=args.minimum_instance_area,
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    history = []
    best_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_or_validate(model, train_loader, device, optimizer)
        val_metrics = train_or_validate(model, val_loader, device)
        row = {
            "epoch": epoch,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))
        payload = {
            "state_dict": model.state_dict(),
            "epoch": epoch,
            "train_args": vars(args),
        }
        torch.save(payload, output / "latest.pth")
        if val_metrics["loss"] < best_loss:
            best_loss = val_metrics["loss"]
            torch.save(payload, output / "best.pth")
            (output / "best_metrics.json").write_text(
                json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        with (output / "history.csv").open("w", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(history[0].keys()))
            writer.writeheader()
            writer.writerows(history)


if __name__ == "__main__":
    main()

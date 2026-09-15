"""区域适应性微调脚本：支持普通微调与小波频域增强微调。"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from boundary_supervision import boundary_supervision_loss
from finetune_manual_tiles import (
    evaluate_samples,
    focal_cross_entropy,
    load_split_file,
    make_train_loader,
    seed_everything,
    target_dice_loss,
    target_tversky_loss,
)
from model.FreqTuneRS3Mamba import FreqTuneRS3Mamba, load_base_rs3mamba_weights
from test_manual_tiles import (
    BANDS_PER_STEP,
    IGNORE_LABEL,
    IN_CHANNELS,
    N_CLASSES,
    TIME_STEPS,
    build_model,
    load_checkpoint,
)
from threshold_sweep_manual_tiles import make_thresholds


DEFAULT_CONFIG = {
    "use_frequency_enhance": True,
    "use_phenology_fusion": False,
    "train_temporal_fusion": False,
    "phenology_fusion_scale": 1.0,
    "frequency_module": "wavelet",
    "freeze_backbone": True,
    "trainable_backbone_prefixes": [],
    "backbone_lr": 1e-5,
    "head_lr": 1e-4,
    "frequency_channels": 64,
    "pretrained_weight": "results_shixun/RS3Mamba_epoch25_miou0.8634.pth",
    "split_file": "manual_tiles_filtered_problem_removed/filtered_trainval_split.json",
    "data_root": r"D:\s2_output\manual_tiles_maize30_stride128",
    "output_dir": "manual_tiles_finetune_frequency_wavelet",
    "epochs": 10,
    "batch_size": 2,
    "val_batch_size": 4,
    "train_window_size": 256,
    "num_workers": 0,
    "weight_decay": 1e-4,
    "class_weights": [1.0, 2.0],
    "dice_weight": 0.5,
    "focal_gamma": 0.0,
    "tversky_weight": 0.0,
    "use_boundary_supervision": False,
    "boundary_radius": 1,
    "boundary_loss_weight": 0.0,
    "boundary_classification_weight": 0.0,
    "boundary_positive_weight": 4.0,
    "boundary_dice_weight": 0.0,
    "stride": 128,
    "threshold_start": 0.40,
    "threshold_end": 0.85,
    "threshold_step": 0.025,
    "max_val_samples": 0,
    "max_train_samples": 0,
    "seed": 42,
    "device": "cuda",
}


def train_one_epoch_adaptation(net, loader, optimizer, device, class_weights, config):
    """兼容原损失，并在启用时增加真实 Mask 边界监督。"""
    net.train()
    totals = {"loss": 0.0, "boundary": 0.0, "boundary_classification": 0.0}
    batches = 0
    for batch in tqdm(loader, desc="train", leave=False):
        image = batch["image"].to(device, non_blocking=True)
        label = batch["label"].to(device, non_blocking=True)
        positions = batch["positions"].to(device, non_blocking=True)
        quality = batch["quality"].to(device, non_blocking=True)
        pad_mask = batch["pad_mask"].to(device, non_blocking=True)

        logits = net(image, batch_positions=positions, quality_score=quality, pad_mask=pad_mask)
        focal_gamma = float(config["focal_gamma"])
        if focal_gamma > 0:
            ce = focal_cross_entropy(logits, label, class_weights, focal_gamma)
        else:
            ce = F.cross_entropy(logits, label, weight=class_weights, ignore_index=IGNORE_LABEL)
        loss = (
            ce
            + float(config["dice_weight"]) * target_dice_loss(logits, label)
            + float(config["tversky_weight"]) * target_tversky_loss(logits, label)
        )

        boundary = logits.sum() * 0.0
        boundary_classification = logits.sum() * 0.0
        if config.get("use_boundary_supervision", False):
            boundary, boundary_classification = boundary_supervision_loss(
                logits,
                label,
                ignore_label=IGNORE_LABEL,
                radius=int(config["boundary_radius"]),
                positive_weight=float(config["boundary_positive_weight"]),
                dice_weight=float(config.get("boundary_dice_weight", 0.0)),
            )
            loss = (
                loss
                + float(config["boundary_loss_weight"]) * boundary
                + float(config["boundary_classification_weight"]) * boundary_classification
            )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optimizer.step()
        totals["loss"] += float(loss.detach().cpu())
        totals["boundary"] += float(boundary.detach().cpu())
        totals["boundary_classification"] += float(boundary_classification.detach().cpu())
        batches += 1
    return {key: value / max(batches, 1) for key, value in totals.items()}


def load_config(path):
    config = dict(DEFAULT_CONFIG)
    if path:
        with Path(path).open("r", encoding="utf-8") as file:
            config.update(json.load(file))
    return config


def build_adaptation_model(config, device):
    if not config["use_frequency_enhance"]:
        model = build_model(device)
        load_checkpoint(model, config["pretrained_weight"], device)
        print("[INFO] 使用普通 RS3Mamba 区域适应性微调。")
        return model

    model = FreqTuneRS3Mamba(
        num_classes=N_CLASSES,
        in_channels=IN_CHANNELS,
        pretrained=False,
        use_phenology_fusion=bool(config.get("use_phenology_fusion", False)),
        time_steps=TIME_STEPS,
        bands_per_step=BANDS_PER_STEP,
        phenology_prior_mode="data",
        phenology_fusion_scale=float(config.get("phenology_fusion_scale", 1.0)),
        use_frequency_enhance=True,
        frequency_module=config["frequency_module"],
        frequency_channels=int(config.get("frequency_channels", 64)),
    ).to(device)
    load_base_rs3mamba_weights(model, config["pretrained_weight"], map_location=device)
    return model


def configure_optimizer(model, config):
    """backbone 使用小学习率，新增频域模块、融合层和解码器使用大学习率。"""
    # Fuse 与 decoder 都属于区域适应阶段的任务头，使用较大学习率更新。
    head_prefixes = ["frequency_enhance.", "frequency_fusion.", "Fuse.", "decoder."]
    if config.get("train_temporal_fusion", False):
        head_prefixes.append("temporal_fusion.")
    head_prefixes = tuple(head_prefixes)
    backbone_prefixes = tuple(config.get("trainable_backbone_prefixes", []))
    backbone_params = []
    head_params = []
    frozen = []
    for name, parameter in model.named_parameters():
        if name.startswith(head_prefixes):
            parameter.requires_grad = True
            head_params.append(parameter)
        elif config["freeze_backbone"] and not name.startswith(backbone_prefixes):
            parameter.requires_grad = False
            frozen.append(name)
        else:
            parameter.requires_grad = True
            backbone_params.append(parameter)

    groups = []
    if backbone_params:
        groups.append(
            {
                "params": backbone_params,
                "lr": float(config["backbone_lr"]),
                "group_name": "backbone",
            }
        )
    if head_params:
        groups.append(
            {
                "params": head_params,
                "lr": float(config["head_lr"]),
                "group_name": "frequency_and_decoder",
            }
        )
    optimizer = torch.optim.AdamW(groups, weight_decay=float(config["weight_decay"]))
    print(
        f"[INFO] 可训练 backbone 参数量: "
        f"{sum(p.numel() for p in backbone_params):,}, lr={config['backbone_lr']}"
    )
    print(
        f"[INFO] 可训练频域/融合/解码器参数量: "
        f"{sum(p.numel() for p in head_params):,}, lr={config['head_lr']}"
    )
    print(f"[INFO] 冻结参数张量数: {len(frozen)}")
    return optimizer


def checkpoint_payload(model, config, epoch, best):
    return {
        "state_dict": model.state_dict(),
        "model_config": {
            "model_type": "FreqTuneRS3Mamba"
            if config["use_frequency_enhance"]
            else "RS3Mamba",
            "use_frequency_enhance": config["use_frequency_enhance"],
            "use_phenology_fusion": config.get("use_phenology_fusion", False),
            "train_temporal_fusion": config.get("train_temporal_fusion", False),
            "phenology_fusion_scale": config.get("phenology_fusion_scale", 1.0),
            "frequency_module": config["frequency_module"],
            "in_channels": IN_CHANNELS,
            "num_classes": N_CLASSES,
        },
        "train_config": config,
        "epoch": epoch,
        "best": best,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/frequency_tune_wavelet.json")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.device:
        config["device"] = args.device

    seed_everything(int(config["seed"]))
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "resolved_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    device_name = config["device"]
    if device_name == "cuda" and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)
    train_samples, val_samples = load_split_file(config["split_file"], config["data_root"])
    if int(config.get("max_train_samples", 0)) > 0:
        train_samples = train_samples[: int(config["max_train_samples"])]
    model = build_adaptation_model(config, device)
    optimizer = configure_optimizer(model, config)
    loader_args = argparse.Namespace(
        batch_size=int(config["batch_size"]),
        num_workers=int(config["num_workers"]),
        window_size=(
            int(config.get("train_window_size", 256)),
            int(config.get("train_window_size", 256)),
        ),
    )
    train_loader = make_train_loader(train_samples, loader_args, device)
    thresholds = make_thresholds(
        float(config["threshold_start"]),
        float(config["threshold_end"]),
        float(config["threshold_step"]),
    )
    class_weights = torch.tensor(config["class_weights"], dtype=torch.float32, device=device)

    history = []
    best_miou = -1.0
    for epoch in range(1, int(config["epochs"]) + 1):
        losses = train_one_epoch_adaptation(model, train_loader, optimizer, device, class_weights, config)
        best, _, skipped = evaluate_samples(
            model,
            val_samples,
            device,
            int(config["stride"]),
            int(config["val_batch_size"]),
            thresholds,
            int(config["max_val_samples"]),
        )
        row = {
            "epoch": epoch,
            "train_loss": losses["loss"],
            "boundary_loss": losses["boundary"],
            "boundary_classification_loss": losses["boundary_classification"],
            "threshold": best["threshold"],
            "oa": best["oa"],
            "target_recall": best["class_acc"][1],
            "target_f1": best["f1"][1],
            "target_iou": best["iou"][1],
            "miou": best["miou"],
            "skipped": len(skipped),
        }
        history.append(row)
        print(
            f"[EPOCH {epoch}] loss={losses['loss']:.4f} boundary={losses['boundary']:.4f} "
            f"MIoU={best['miou']:.4f} "
            f"targetIoU={best['iou'][1]:.4f} F1={best['f1'][1]:.4f} "
            f"threshold={best['threshold']:.3f}"
        )
        torch.save(checkpoint_payload(model, config, epoch, best), output_dir / "latest.pth")
        if best["miou"] > best_miou:
            best_miou = best["miou"]
            torch.save(checkpoint_payload(model, config, epoch, best), output_dir / "best.pth")
            (output_dir / "best_metrics.json").write_text(
                json.dumps({"epoch": epoch, "best": best, "skipped": skipped}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        with (output_dir / "history.csv").open("w", newline="", encoding="utf-8-sig") as file:
            writer = csv.DictWriter(file, fieldnames=history[0].keys())
            writer.writeheader()
            writer.writerows(history)

    print(f"[INFO] 训练完成，最佳验证集 MIoU={best_miou:.4f}")


if __name__ == "__main__":
    main()

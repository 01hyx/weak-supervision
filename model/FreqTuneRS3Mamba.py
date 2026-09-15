"""区域适应性微调阶段使用的小波频域增强 RS3Mamba。"""

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.RS3Mamba import RS3Mamba


class HaarWaveletDecomposition(nn.Module):
    """无参数 Haar 小波分解，输出 LL、LH、HL、HH 四个分量。"""

    def forward(self, x):
        # 奇数尺寸先补齐，确保 2x2 下采样可以完整覆盖。
        pad_h = x.shape[-2] % 2
        pad_w = x.shape[-1] % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")

        x00 = x[:, :, 0::2, 0::2]
        x01 = x[:, :, 0::2, 1::2]
        x10 = x[:, :, 1::2, 0::2]
        x11 = x[:, :, 1::2, 1::2]

        ll = (x00 + x01 + x10 + x11) * 0.5
        lh = (-x00 - x01 + x10 + x11) * 0.5
        hl = (-x00 + x01 - x10 + x11) * 0.5
        hh = (x00 - x01 - x10 + x11) * 0.5
        return ll, lh, hl, hh


class WaveletFrequencyEnhance(nn.Module):
    """拼接四个 Haar 分量，并通过 1x1 Conv + BN + ReLU 完成降维整合。"""

    def __init__(self, in_channels=36, out_channels=64):
        super().__init__()
        self.haar = HaarWaveletDecomposition()
        self.reduce = nn.Sequential(
            nn.Conv2d(in_channels * 4, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        ll, lh, hl, hh = self.haar(x)
        return self.reduce(torch.cat([ll, lh, hl, hh], dim=1))


class SpatialFrequencyFusion(nn.Module):
    """使用 concat + 1x1 Conv + 3x3 Conv 融合空间语义与频域特征。"""

    def __init__(self, spatial_channels=64, frequency_channels=64):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(
                spatial_channels + frequency_channels,
                spatial_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(spatial_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                spatial_channels,
                spatial_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(spatial_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, spatial, frequency):
        if frequency.shape[-2:] != spatial.shape[-2:]:
            frequency = F.interpolate(
                frequency,
                size=spatial.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        return self.fuse(torch.cat([spatial, frequency], dim=1))


class FreqTuneRS3Mamba(RS3Mamba):
    """仅在区域适应性微调阶段启用频域增强的 RS3Mamba。"""

    def __init__(
        self,
        *args,
        use_frequency_enhance=True,
        frequency_module="wavelet",
        frequency_channels=64,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if frequency_module != "wavelet":
            raise ValueError(f"Unsupported frequency_module: {frequency_module}")

        self.use_frequency_enhance = use_frequency_enhance
        self.frequency_module_name = frequency_module
        if self.use_frequency_enhance:
            in_channels = self.conv1.in_channels
            shallow_channels = self.backbone.feature_info.channels()[0]
            self.frequency_enhance = WaveletFrequencyEnhance(
                in_channels=in_channels,
                out_channels=frequency_channels,
            )
            self.frequency_fusion = SpatialFrequencyFusion(
                spatial_channels=shallow_channels,
                frequency_channels=frequency_channels,
            )

    def forward(self, x, batch_positions=None, quality_score=None, pad_mask=None):
        if self.use_phenology_fusion:
            x = self.temporal_fusion(
                x,
                batch_positions=batch_positions,
                quality_score=quality_score,
                pad_mask=pad_mask,
            )

        h, w = x.shape[-2:]
        frequency = self.frequency_enhance(x) if self.use_frequency_enhance else None

        ssmx = self.stem(x)
        vss_outs = self.vssm_encoder(ssmx)

        ress = []
        spatial = self.conv1(x)
        spatial = self.bn1(spatial)
        spatial = self.act1(spatial)
        spatial = self.maxpool(spatial)
        for index, layer in enumerate(self.layers):
            spatial = layer(spatial)
            spatial = self.Fuse[index](spatial, vss_outs[index + 1])
            if index == 0 and frequency is not None:
                spatial = self.frequency_fusion(spatial, frequency)
            ress.append(spatial)

        return self.decoder(ress[0], ress[1], ress[2], ress[3], h, w)


def load_base_rs3mamba_weights(model, checkpoint_path, map_location="cpu"):
    """加载原始 RS3Mamba 权重，并明确打印继承参数与新增初始化参数。"""
    checkpoint_path = Path(checkpoint_path)
    state = torch.load(checkpoint_path, map_location=map_location)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    # 兼容 DataParallel 保存的 module. 前缀。
    state = {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in state.items()
    }
    model_state = model.state_dict()
    compatible = {
        key: value
        for key, value in state.items()
        if key in model_state and model_state[key].shape == value.shape
    }
    incompatible_shape = [
        key
        for key, value in state.items()
        if key in model_state and model_state[key].shape != value.shape
    ]
    missing, unexpected = model.load_state_dict(compatible, strict=False)

    inherited = sorted(compatible)
    new_parameters = sorted(
        key
        for key in missing
        if key.startswith(("frequency_enhance.", "frequency_fusion."))
    )
    other_missing = sorted(set(missing) - set(new_parameters))
    truly_unexpected = sorted(set(state) - set(model_state))

    print(f"[INFO] 预训练权重: {checkpoint_path}")
    print(f"[INFO] 成功继承参数: {len(inherited)}")
    print(f"[INFO] 新增频域模块参数（随机初始化）: {len(new_parameters)}")
    if new_parameters:
        for key in new_parameters:
            print(f"  [NEW] {key}")
    if other_missing:
        print(f"[WARN] 其他缺失参数: {len(other_missing)}")
        for key in other_missing[:20]:
            print(f"  [MISSING] {key}")
    if truly_unexpected or unexpected:
        keys = sorted(set(truly_unexpected) | set(unexpected))
        print(f"[WARN] 未使用的权重参数: {len(keys)}")
        for key in keys[:20]:
            print(f"  [UNEXPECTED] {key}")
    if incompatible_shape:
        print(f"[WARN] 形状不匹配参数: {len(incompatible_shape)}")
        for key in incompatible_shape[:20]:
            print(f"  [SHAPE] {key}")
    return {
        "loaded": inherited,
        "new_parameters": new_parameters,
        "missing": other_missing,
        "unexpected": sorted(set(truly_unexpected) | set(unexpected)),
        "shape_mismatch": incompatible_shape,
    }

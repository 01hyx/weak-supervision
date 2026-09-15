"""RS3Mamba with semantic, parcel-boundary, center and offset heads."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.FreqTuneRS3Mamba import FreqTuneRS3Mamba


def prediction_head(channels, output_channels):
    return nn.Sequential(
        nn.Conv2d(channels, channels, 3, padding=1, bias=False),
        nn.BatchNorm2d(channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(channels, output_channels, 1),
    )


class ParcelInstanceRS3Mamba(FreqTuneRS3Mamba):
    """在原有语义分割解码特征上增加地块实例预测头。"""

    def __init__(self, *args, instance_channels=64, **kwargs):
        super().__init__(*args, **kwargs)
        self.instance_refine = nn.Sequential(
            nn.Conv2d(instance_channels * 2, instance_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(instance_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(instance_channels, instance_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(instance_channels),
            nn.ReLU(inplace=True),
        )
        self.parcel_boundary_head = prediction_head(instance_channels, 1)
        self.parcel_center_head = prediction_head(instance_channels, 1)
        self.parcel_offset_head = prediction_head(instance_channels, 2)

    def decode_features(self, features):
        decoder = self.decoder
        x = decoder.b4(decoder.pre_conv(features[3]))
        x = decoder.p3(x, features[2])
        x = decoder.b3(x)
        x = decoder.p2(x, features[1])
        x = decoder.b2(x)
        return decoder.p1(x, features[0])

    def forward(self, x, batch_positions=None, quality_score=None, pad_mask=None):
        if self.use_phenology_fusion:
            x = self.temporal_fusion(
                x,
                batch_positions=batch_positions,
                quality_score=quality_score,
                pad_mask=pad_mask,
            )

        output_size = x.shape[-2:]
        frequency = self.frequency_enhance(x) if self.use_frequency_enhance else None
        ssmx = self.stem(x)
        vss_outs = self.vssm_encoder(ssmx)

        features = []
        spatial = self.act1(self.bn1(self.conv1(x)))
        spatial = self.maxpool(spatial)
        for index, layer in enumerate(self.layers):
            spatial = layer(spatial)
            spatial = self.Fuse[index](spatial, vss_outs[index + 1])
            if index == 0 and frequency is not None:
                spatial = self.frequency_fusion(spatial, frequency)
            features.append(spatial)

        decoded = self.decode_features(features)
        semantic = self.decoder.segmentation_head(decoded)
        # 实例头在 1/2 分辨率融合解码语义与 WFEM 细节，减少小地块中心被 1/4 特征平滑的问题。
        decoded_high = F.interpolate(
            decoded, size=frequency.shape[-2:], mode="bilinear", align_corners=False
        )
        instance_feature = self.instance_refine(torch.cat([decoded_high, frequency], dim=1))
        boundary = self.parcel_boundary_head(instance_feature)
        center = self.parcel_center_head(instance_feature)
        offset = self.parcel_offset_head(instance_feature)
        outputs = {
            "semantic": semantic,
            "boundary": boundary,
            "center": center,
            "offset": offset,
        }
        return {
            key: F.interpolate(value, size=output_size, mode="bilinear", align_corners=False)
            for key, value in outputs.items()
        }

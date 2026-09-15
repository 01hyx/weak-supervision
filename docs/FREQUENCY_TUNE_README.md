# 小波频域增强区域适应性微调

本实现以新增模型和脚本的方式接入频域增强，不修改原始
`model/RS3Mamba.py`、原始训练脚本及原始推理流程。

## 模型结构

`FreqTuneRS3Mamba` 继承原始 `RS3Mamba`：

1. 保持六期 Sentinel-2 的 36 通道输入不变。
2. 对输入执行 Haar 小波分解，得到 LL、LH、HL、HH。
3. 拼接四个频域分量，并通过 `1x1 Conv + BN + ReLU` 整合。
4. 将频域特征插值至浅层空间特征尺寸。
5. 使用 `concat + 1x1 Conv + 3x3 Conv` 融合后送入原解码流程。

原始预训练权重使用兼容加载方式继承；新增频域模块和融合层随机初始化。

## 三组实验

1. 基础 RS3Mamba 直接应用：

```powershell
python evaluate_adaptation_experiments.py --experiment-configs configs/experiment_base_direct.json
```

2. 区域适应性微调，不加频域：

```powershell
python finetune_frequency_adaptation.py --config configs/adaptation_no_frequency.json
```

3. 小波频域增强区域适应性微调：

```powershell
python finetune_frequency_adaptation.py --config configs/frequency_tune_wavelet.json
```

完成两组微调后，统一评估三组实验：

```powershell
python evaluate_adaptation_experiments.py `
  --experiment-configs configs/experiment_base_direct.json configs/experiment_adaptation_no_frequency.json configs/experiment_frequency_wavelet.json `
  --save-boundary-overlays
```

评估输出包括 OA、Precision、Recall、F1、玉米 IoU、MIoU，以及预测斑块数、
小连通域数量、平均斑块面积和边界叠加图。

## 推理导出

```powershell
python export_frequency_tune_predictions.py --config configs/experiment_frequency_wavelet.json
```

推理结果包括概率 GeoTIFF 和二值 GeoTIFF，不改变原始数据读取格式。

## 关键配置

```json
{
  "use_frequency_enhance": true,
  "frequency_module": "wavelet",
  "freeze_backbone": true,
  "backbone_lr": 1e-5,
  "head_lr": 1e-4,
  "pretrained_weight": "results_shixun/RS3Mamba_epoch25_miou0.8634.pth"
}
```

`freeze_backbone=true` 时，主干冻结，频域模块、融合层、原 `Fuse` 层和解码器
使用 `head_lr` 更新；设为 `false` 时，主干使用较小的 `backbone_lr` 联合微调。

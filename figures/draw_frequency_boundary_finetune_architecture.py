"""绘制频域增强与边界监督区域微调模型结构图。"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


FONT = font_manager.FontProperties(fname=r"C:\Windows\Fonts\msyh.ttc")
FONT_BOLD = font_manager.FontProperties(fname=r"C:\Windows\Fonts\msyhbd.ttc")

COLORS = {
    "frozen": "#BFDDF0",
    "train": "#F4A261",
    "decoder": "#DCC8E8",
    "frequency": "#F6D98B",
    "loss": "#F3C5C5",
    "green": "#5BB98C",
    "line": "#263238",
    "panel": "#FFFFFF",
    "soft_blue": "#EDF6FB",
    "soft_orange": "#FFF2DF",
    "soft_purple": "#F7EFFA",
    "soft_green": "#EEF8EC",
}


def text(ax, x, y, value, size=10, bold=False, ha="center", va="center", rotation=0):
    ax.text(
        x, y, value, fontsize=size, fontproperties=FONT_BOLD if bold else FONT,
        ha=ha, va=va, color="#202020", rotation=rotation, zorder=10
    )


def node(ax, x, y, w, h, label, color, size=9, radius=0.015, linewidth=1.4, dashed=False):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.004,rounding_size={radius}",
        facecolor=color, edgecolor=COLORS["line"], linewidth=linewidth,
        linestyle="--" if dashed else "-", zorder=3
    )
    ax.add_patch(patch)
    text(ax, x + w / 2, y + h / 2, label, size=size, bold=True)
    return patch


def panel(ax, x, y, w, h, label=None, color="#FFFFFF", dashed=False, radius=0.015):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.006,rounding_size={radius}",
        facecolor=color, edgecolor=COLORS["line"], linewidth=1.5,
        linestyle="--" if dashed else "-", zorder=1
    )
    ax.add_patch(patch)
    if label:
        text(ax, x + 0.012, y + h - 0.018, label, size=10, bold=True, ha="left", va="top")
    return patch


def arrow(ax, x1, y1, x2, y2, color="#263238", dashed=False, width=1.5, style="-|>"):
    patch = FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle=style, mutation_scale=12,
        linewidth=width, color=color, linestyle="--" if dashed else "-",
        connectionstyle="arc3,rad=0", zorder=5
    )
    ax.add_patch(patch)
    return patch


def line(ax, xs, ys, color="#263238", dashed=False, width=1.4):
    ax.plot(xs, ys, color=color, linestyle="--" if dashed else "-", linewidth=width, zorder=4)


def feature_stack(ax, x, y, w, h, count, color, label):
    for i in range(count - 1, -1, -1):
        offset = i * 0.006
        ax.add_patch(Rectangle(
            (x + offset, y + offset), w, h, facecolor=color,
            edgecolor=COLORS["line"], linewidth=1.0, zorder=3 + i
        ))
    text(ax, x + w / 2 + 0.008, y + h / 2 + 0.008, label, size=8, bold=True)


def draw_main(ax):
    panel(ax, 0.015, 0.385, 0.97, 0.60, color="#FFFFFF", radius=0.004)
    text(ax, 0.5, 0.958, "频域增强与边界监督区域适应性微调模型", size=16, bold=True)

    # Input stack
    feature_stack(ax, 0.045, 0.685, 0.065, 0.13, 5, "#B9D9C4", "六期 Sentinel-2\n36 通道输入")
    text(ax, 0.084, 0.645, "X ∈ R(36×H×W)", size=8)

    # Frozen backbone
    panel(ax, 0.16, 0.585, 0.41, 0.29, "预训练 RS3Mamba 主干（冻结）", COLORS["soft_blue"], radius=0.02)
    node(ax, 0.185, 0.71, 0.07, 0.07, "Stem", COLORS["frozen"])
    node(ax, 0.285, 0.71, 0.09, 0.07, "双分支编码器\nStage 1", COLORS["frozen"], size=8)
    node(ax, 0.405, 0.71, 0.065, 0.07, "Stage\n2–4", COLORS["frozen"], size=8)
    node(ax, 0.495, 0.71, 0.055, 0.07, "多尺度\n特征", COLORS["frozen"], size=8)
    arrow(ax, 0.255, 0.745, 0.285, 0.745)
    arrow(ax, 0.375, 0.745, 0.405, 0.745)
    arrow(ax, 0.47, 0.745, 0.495, 0.745)
    text(ax, 0.33, 0.825, "空间—时序语义特征", size=9, bold=True)

    # Frequency branch
    panel(ax, 0.16, 0.42, 0.41, 0.135, "新增小波频域分支（可训练）", COLORS["soft_orange"], radius=0.02)
    node(ax, 0.185, 0.455, 0.085, 0.055, "Haar 小波\n分解", COLORS["frequency"], size=8)
    node(ax, 0.30, 0.455, 0.09, 0.055, "LL / LH /\nHL / HH", "#FFE5A8", size=8)
    node(ax, 0.42, 0.455, 0.125, 0.055, "Concat + 1×1 Conv\nBN + ReLU", COLORS["train"], size=8)
    arrow(ax, 0.27, 0.482, 0.30, 0.482)
    arrow(ax, 0.39, 0.482, 0.42, 0.482)

    # Input split arrows
    arrow(ax, 0.12, 0.755, 0.185, 0.745)
    line(ax, [0.135, 0.135, 0.185], [0.755, 0.482, 0.482])
    arrow(ax, 0.17, 0.482, 0.185, 0.482)

    # Fusion and decoder
    node(ax, 0.61, 0.67, 0.105, 0.105, "空间—频域\n特征融合", COLORS["train"], size=9)
    arrow(ax, 0.55, 0.745, 0.61, 0.725, color=COLORS["green"], width=2.0)
    line(ax, [0.545, 0.585, 0.585, 0.61], [0.482, 0.482, 0.70, 0.70], color=COLORS["green"], width=2.0)
    arrow(ax, 0.595, 0.70, 0.61, 0.70, color=COLORS["green"], width=2.0)

    node(ax, 0.755, 0.67, 0.09, 0.105, "解码器", COLORS["decoder"], size=10)
    node(ax, 0.885, 0.67, 0.075, 0.105, "玉米概率图\nP", "#C8E6C9", size=9)
    arrow(ax, 0.715, 0.722, 0.755, 0.722)
    arrow(ax, 0.845, 0.722, 0.885, 0.722)

    # Boundary supervision
    panel(ax, 0.61, 0.42, 0.35, 0.17, "训练阶段边界监督", COLORS["soft_purple"], dashed=True, radius=0.02)
    node(ax, 0.63, 0.465, 0.075, 0.065, "真实 Mask\nY", "#FFFFFF", size=8)
    node(ax, 0.735, 0.465, 0.10, 0.065, "形态学梯度\n真实边界", COLORS["loss"], size=8)
    node(ax, 0.865, 0.465, 0.075, 0.065, "边界损失\nL_boundary", COLORS["loss"], size=7)
    arrow(ax, 0.705, 0.497, 0.735, 0.497)
    arrow(ax, 0.835, 0.497, 0.865, 0.497)
    line(ax, [0.922, 0.922, 0.902], [0.67, 0.56, 0.56], dashed=True, color="#777777")
    arrow(ax, 0.902, 0.56, 0.902, 0.53, dashed=True, color="#777777")
    text(ax, 0.765, 0.435, "仅用于微调训练，不改变推理结构", size=8)

    # Legend
    panel(ax, 0.60, 0.80, 0.36, 0.115, color="#FFFFFF", dashed=True, radius=0.012)
    ax.scatter([0.625], [0.872], s=70, color=COLORS["frozen"], edgecolors=COLORS["line"], zorder=6)
    text(ax, 0.645, 0.872, "继承预训练权重并冻结", size=8, ha="left")
    ax.scatter([0.625], [0.835], s=70, color=COLORS["train"], edgecolors=COLORS["line"], zorder=6)
    text(ax, 0.645, 0.835, "区域微调阶段可训练模块", size=8, ha="left")
    line(ax, [0.79, 0.825], [0.872, 0.872], color=COLORS["green"], width=2.2)
    text(ax, 0.84, 0.872, "频域特征路径", size=8, ha="left")
    line(ax, [0.79, 0.825], [0.835, 0.835], color="#777777", dashed=True, width=1.5)
    text(ax, 0.84, 0.835, "监督损失路径", size=8, ha="left")

    text(ax, 0.5, 0.398, "(a) 频域增强与边界监督区域微调总体框架", size=10, bold=True, va="bottom")


def draw_wavelet(ax):
    panel(ax, 0.015, 0.035, 0.31, 0.325, color="#FFFFFF", radius=0.004)
    text(ax, 0.17, 0.342, "(b) Haar 小波频域增强模块", size=10, bold=True)
    node(ax, 0.04, 0.165, 0.055, 0.07, "输入\nX", "#B9D9C4", size=8)
    node(ax, 0.12, 0.165, 0.07, 0.07, "Haar\n分解", COLORS["frequency"], size=8)
    arrow(ax, 0.095, 0.20, 0.12, 0.20)
    ys = [0.265, 0.205, 0.145, 0.085]
    labels = ["LL 低频结构", "LH 水平细节", "HL 垂直细节", "HH 对角细节"]
    colors = ["#B7D7A8", "#FFD59A", "#F8C6A8", "#E7B5C7"]
    for y, label, color in zip(ys, labels, colors):
        node(ax, 0.215, y, 0.085, 0.042, label, color, size=6.7, radius=0.008)
        arrow(ax, 0.19, 0.20, 0.215, y + 0.021)
    node(ax, 0.105, 0.065, 0.085, 0.045, "通道拼接", "#FFE5A8", size=7)
    node(ax, 0.215, 0.065, 0.085, 0.045, "1×1 Conv\nBN + ReLU", COLORS["train"], size=7)
    arrow(ax, 0.19, 0.087, 0.215, 0.087)
    for y in ys:
        line(ax, [0.257, 0.257, 0.19], [y, 0.125, 0.125])
    arrow(ax, 0.19, 0.125, 0.148, 0.11)


def draw_fusion(ax):
    panel(ax, 0.345, 0.035, 0.31, 0.325, color="#FFFFFF", radius=0.004)
    text(ax, 0.50, 0.342, "(c) 空间—频域特征融合模块", size=10, bold=True)
    node(ax, 0.37, 0.235, 0.075, 0.055, "空间特征\nF_spatial", COLORS["frozen"], size=7)
    node(ax, 0.37, 0.105, 0.075, 0.055, "频域特征\nF_freq", COLORS["frequency"], size=7)
    node(ax, 0.47, 0.105, 0.075, 0.055, "双线性插值\n尺寸对齐", "#FFE5A8", size=7)
    node(ax, 0.47, 0.195, 0.075, 0.055, "Concat", COLORS["train"], size=8)
    node(ax, 0.57, 0.195, 0.06, 0.055, "1×1\nConv", COLORS["train"], size=8)
    node(ax, 0.57, 0.105, 0.06, 0.055, "3×3\nConv", COLORS["train"], size=8)
    arrow(ax, 0.445, 0.262, 0.47, 0.222)
    arrow(ax, 0.445, 0.132, 0.47, 0.132)
    arrow(ax, 0.507, 0.16, 0.507, 0.195)
    arrow(ax, 0.545, 0.222, 0.57, 0.222)
    arrow(ax, 0.60, 0.195, 0.60, 0.16)
    text(ax, 0.60, 0.075, "融合特征 F_fused", size=8, bold=True)
    arrow(ax, 0.60, 0.105, 0.60, 0.087)


def draw_loss(ax):
    panel(ax, 0.675, 0.035, 0.31, 0.325, color="#FFFFFF", radius=0.004)
    text(ax, 0.83, 0.342, "(d) 边界监督损失", size=10, bold=True)
    node(ax, 0.70, 0.235, 0.065, 0.055, "真实 Mask", "#FFFFFF", size=7)
    node(ax, 0.70, 0.115, 0.065, 0.055, "概率图 P", "#C8E6C9", size=7)
    node(ax, 0.79, 0.235, 0.075, 0.055, "形态学梯度", COLORS["loss"], size=7)
    node(ax, 0.79, 0.115, 0.075, 0.055, "可微形态学\n梯度", COLORS["loss"], size=7)
    node(ax, 0.89, 0.195, 0.065, 0.055, "边界 BCE", COLORS["loss"], size=7)
    arrow(ax, 0.765, 0.262, 0.79, 0.262)
    arrow(ax, 0.765, 0.142, 0.79, 0.142)
    arrow(ax, 0.865, 0.262, 0.89, 0.225)
    arrow(ax, 0.865, 0.142, 0.89, 0.215)
    text(ax, 0.83, 0.072, "L = L_CE + λd L_Dice + λb L_Boundary + λc L_BoundaryClass", size=7.0, bold=True)


def main():
    output = Path("paper_figures")
    output.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(16, 10), dpi=300, facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    draw_main(ax)
    draw_wavelet(ax)
    draw_fusion(ax)
    draw_loss(ax)
    png = output / "frequency_boundary_finetune_architecture.png"
    svg = output / "frequency_boundary_finetune_architecture.svg"
    fig.savefig(png, dpi=300, bbox_inches="tight", pad_inches=0.03, facecolor="white")
    fig.savefig(svg, bbox_inches="tight", pad_inches=0.03, facecolor="white")
    plt.close(fig)
    print(png.resolve())
    print(svg.resolve())


if __name__ == "__main__":
    main()

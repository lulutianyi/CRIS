"""
【ATTN 定义之后的新代码｜FlowPure 在 D-Fire 上的训练与评测】

这个文件不是独立程序。它只保留 gcn.ipynb 的 ATTN 定义单元之后新增的
FlowPure 部分，并直接复用当前 Colab 会话中已经存在的：

    bld.DFIRE_ROOT          D-Fire 根目录（你的数据位于 /content）
    DFireDetDataset         notebook 已定义的数据集
    det_collate             notebook 已定义的检测 batch 拼接函数
    attack_yolo, criterion  notebook 已加载的 PGD 攻击模型与检测损失
    detector                notebook 已加载的 mAP 检测器
    save_as_dfire_split     notebook 已定义的图像/标签保存函数
    eval_map                notebook 已定义的 mAP 函数

正确载入方式（必须保留 -i，共享 notebook 内存）：

    %run -i /content/flowpure_dfire.py

不要再使用：

    !python /content/flowpure_dfire.py ...

因为 !python 会启动一个看不到 notebook 现有变量的新进程。

本文件实现的是 FlowPure-PGD 的 D-Fire 检测适配：训练时用冻结 YOLO 生成
PGD 图像，并通过 deterministic conditional flow matching 学习
“对抗图 -> 干净图”的速度场；评测为攻击者不知道净化器的 realistic/
preprocessor-blind 协议。
"""

from __future__ import annotations

import json
import math
import os
import random
import time
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


# =============================================================================
# 0. notebook 会话检查：不重新加载环境、数据、YOLO 或 ATTN/GCN
# =============================================================================


_FLOWPURE_REQUIRED_NOTEBOOK_NAMES = (
    "bld",
    "DFireDetDataset",
    "det_collate",
    "attack_yolo",
    "criterion",
    "detector",
    "save_as_dfire_split",
    "eval_map",
)


def flowpure_check_notebook_environment() -> str:
    missing = [
        name for name in _FLOWPURE_REQUIRED_NOTEBOOK_NAMES if name not in globals()
    ]
    if missing:
        raise RuntimeError(
            "FlowPure 扩展无法访问 notebook 中已定义的对象："
            + ", ".join(missing)
            + "。请先按顺序运行 gcn.ipynb 到 ATTN 定义单元，然后使用 "
            + "%run -i /content/flowpure_dfire.py；不要用 !python。"
        )

    dfire_root = str(bld.DFIRE_ROOT)
    print(f"FlowPure 复用当前 notebook 的 D-Fire 路径: {dfire_root}")
    print("FlowPure 复用当前 notebook 的 YOLO、PGD loss、数据集和 mAP 函数")
    return dfire_root


FLOWPURE_DFIRE_ROOT = flowpure_check_notebook_environment()


# =============================================================================
# 1. FlowPure 新配置（这里没有 D-Fire 数据路径和 YOLO 权重路径）
# =============================================================================


@dataclass
class FlowPureModelConfig:
    image_size: int = 128
    in_channels: int = 3
    base_channels: int = 64
    channel_mult: Tuple[int, ...] = (1, 2, 4, 4)
    attention_resolutions: Tuple[int, ...] = (16,)
    dropout: float = 0.1


@dataclass
class FlowPureConfig:
    # Drive 只用于保存训练权重和评测结果，不用于读取 D-Fire。
    checkpoint: str = (
        "/content/drive/MyDrive/flowpure_dfire_checkpoints/flowpure_pgd.pt"
    )
    output_root: str = "/content/drive/MyDrive"

    train_split: str = "train"
    eval_split: str = "test"
    detector_size: int = 384
    flow_size: int = 128
    expected_eval_images: int = 4306
    max_eval_images: Optional[int] = None

    # 论文尺度目标：300k optimizer steps，学习率 2e-4，有效 batch 64。
    max_steps: int = 300_000
    learning_rate: float = 2e-4
    physical_batch_size: int = 4
    gradient_accumulation: int = 16
    num_workers: int = 2
    max_grad_norm: float = 1.0
    snapshot_every: int = 5_000
    log_every: int = 20
    use_amp: bool = True

    # FlowPure-PGD：训练 eps ~ U(0, 0.05)，PGD-10，alpha=2/255。
    train_eps_max: float = 0.05
    eval_eps: float = 8 / 255
    pgd_alpha: float = 2 / 255
    pgd_steps: int = 10

    # 推理时从 t=0 到 t=1 的 Euler 步数（论文默认 10 NFE）。
    euler_steps: int = 10

    eval_batch_size: int = 8
    train_seed: int = 20260826
    attack_seed: int = 20260822

    # 防止不同 Ultralytics 版本的 criterion.hyp 缺少这些字段。
    det_box_gain: float = 7.5
    det_cls_gain: float = 0.5
    det_dfl_gain: float = 1.5


# 载入本文件后可以直接修改这两个对象，再调用训练/评测函数。
flowpure_model_cfg = FlowPureModelConfig()
flowpure_cfg = FlowPureConfig()


# =============================================================================
# 2. FlowPure 新增的 YOLO-PGD 适配
#    复用 notebook 的 attack_yolo/criterion，不重新读取任何权重
# =============================================================================


class _FlowPureLossAttrDict(dict):
    """兼容 Ultralytics 同时使用 hyp['box'] 与 hyp.box 的版本差异。"""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = value


def flowpure_prepare_notebook_attack(cfg: FlowPureConfig) -> None:
    """冻结现有攻击 YOLO，但保留 detection loss 所需的 train 输出格式。"""

    attack_yolo.model.to(bld.DEVICE)
    attack_yolo.model.train()
    for parameter in attack_yolo.model.parameters():
        parameter.requires_grad_(False)
    # model.train() 会打开 BN；这里单独固定 running mean/var。
    for module in attack_yolo.model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()

    defaults = {
        "box": cfg.det_box_gain,
        "cls": cfg.det_cls_gain,
        "dfl": cfg.det_dfl_gain,
    }
    if isinstance(criterion.hyp, dict):
        hyp = _FlowPureLossAttrDict(criterion.hyp)
        for key, value in defaults.items():
            hyp.setdefault(key, value)
        criterion.hyp = hyp
    else:
        for key, value in defaults.items():
            if not hasattr(criterion.hyp, key):
                setattr(criterion.hyp, key, value)

    print(
        "复用并冻结 attack_yolo；loss gains:",
        f"box={criterion.hyp.box}, cls={criterion.hyp.cls}, dfl={criterion.hyp.dfl}",
    )


def flowpure_sanitize_targets(
    targets: Dict[str, torch.Tensor],
    device: torch.device,
    num_classes: int = 2,
) -> Tuple[Dict[str, torch.Tensor], int]:
    """丢弃非有限、越界或类别无效的 YOLO 框，避免 loss 崩溃。"""

    clean = {key: value.to(device) for key, value in targets.items()}
    boxes = clean["bboxes"]
    if boxes.numel() == 0:
        return clean, 0

    classes = clean["cls"].flatten()
    valid = torch.isfinite(boxes).all(dim=1)
    valid &= ((boxes >= 0.0) & (boxes <= 1.0)).all(dim=1)
    valid &= torch.isfinite(classes)
    valid &= (classes >= 0) & (classes < num_classes)
    dropped = int((~valid).sum().item())
    if dropped:
        clean["bboxes"] = clean["bboxes"][valid]
        clean["cls"] = clean["cls"][valid]
        clean["batch_idx"] = clean["batch_idx"][valid]
    return clean, dropped


def _flowpure_eps_tensor(
    eps: float | torch.Tensor, x: torch.Tensor
) -> torch.Tensor:
    if torch.is_tensor(eps):
        eps_tensor = eps.to(device=x.device, dtype=x.dtype)
    else:
        eps_tensor = torch.tensor(float(eps), device=x.device, dtype=x.dtype)
    if eps_tensor.ndim == 0:
        return eps_tensor.view(1, 1, 1, 1)
    if eps_tensor.ndim == 1:
        return eps_tensor.view(-1, 1, 1, 1)
    return eps_tensor


def flowpure_pgd_attack(
    x: torch.Tensor,
    targets: Dict[str, torch.Tensor],
    eps: float | torch.Tensor = 8 / 255,
    alpha: float = 2 / 255,
    steps: int = 10,
) -> torch.Tensor:
    """
    对当前 notebook 的 attack_yolo 做 L-inf PGD。

    与 notebook 旧 pgd_attack 的区别是：eps 可以是每张图一个数，因此训练时
    能真正实现论文的 eps ~ U(0, 0.05)。
    """

    if targets["bboxes"].numel() == 0:
        return x.detach().clone()

    eps_tensor = _flowpure_eps_tensor(eps, x)
    lower = (x - eps_tensor).clamp(0.0, 1.0)
    upper = (x + eps_tensor).clamp(0.0, 1.0)
    x_adv = (
        x + torch.empty_like(x).uniform_(-1.0, 1.0) * eps_tensor
    ).clamp(0.0, 1.0)

    with torch.enable_grad():
        for _ in range(steps):
            x_adv = x_adv.detach().requires_grad_(True)
            predictions = attack_yolo.model(x_adv)
            loss_vector, _ = criterion(predictions, targets)
            loss = loss_vector.sum()
            gradient = torch.autograd.grad(loss, x_adv, only_inputs=True)[0]
            x_adv = x_adv.detach() + float(alpha) * gradient.sign()
            x_adv = torch.maximum(torch.minimum(x_adv, upper), lower)
            x_adv = x_adv.clamp(0.0, 1.0)
    return x_adv.detach()


# =============================================================================
# 3. FlowPure 新模型：带时间条件的 velocity U-Net
# =============================================================================


def _flowpure_group_count(channels: int, maximum: int = 32) -> int:
    groups = min(maximum, channels)
    while channels % groups != 0:
        groups -= 1
    return groups


class FlowPureTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        frequencies = torch.exp(
            -math.log(10_000.0)
            * torch.arange(half, device=t.device, dtype=torch.float32)
            / (half - 1)
        )
        angles = (t.float() * 1000.0)[:, None] * frequencies[None, :]
        embedding = torch.cat([angles.sin(), angles.cos()], dim=1)
        if self.dim % 2:
            embedding = F.pad(embedding, (0, 1))
        return embedding


class FlowPureResidualBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, time_dim: int, dropout: float
    ):
        super().__init__()
        self.norm1 = nn.GroupNorm(_flowpure_group_count(in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.time_projection = nn.Linear(time_dim, out_channels)
        self.norm2 = nn.GroupNorm(_flowpure_group_count(out_channels), out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.skip = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_projection(F.silu(temb))[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)


class FlowPureSelfAttention(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        self.norm = nn.GroupNorm(_flowpure_group_count(channels), channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        q, k, v = self.qkv(self.norm(x)).chunk(3, dim=1)
        q = q.flatten(2).transpose(1, 2)
        k = k.flatten(2)
        v = v.flatten(2).transpose(1, 2)
        attention = torch.softmax(torch.bmm(q, k) * channels**-0.5, dim=-1)
        h = torch.bmm(attention, v).transpose(1, 2)
        h = h.reshape(batch, channels, height, width)
        return x + self.proj(h)


class FlowPureUNet(nn.Module):
    """预测 pixel-space velocity v_theta(t, x_t)。"""

    def __init__(self, cfg: FlowPureModelConfig):
        super().__init__()
        self.cfg = cfg
        time_dim = cfg.base_channels * 4
        self.time_embedding = nn.Sequential(
            FlowPureTimeEmbedding(cfg.base_channels),
            nn.Linear(cfg.base_channels, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.input_conv = nn.Conv2d(
            cfg.in_channels, cfg.base_channels, 3, padding=1
        )

        self.down_levels = nn.ModuleList()
        current_channels = cfg.base_channels
        current_resolution = cfg.image_size
        skip_channels: List[int] = []
        for level_index, multiplier in enumerate(cfg.channel_mult):
            out_channels = cfg.base_channels * multiplier
            attention = (
                FlowPureSelfAttention(out_channels)
                if current_resolution in cfg.attention_resolutions
                else nn.Identity()
            )
            downsample = (
                nn.Conv2d(out_channels, out_channels, 4, stride=2, padding=1)
                if level_index < len(cfg.channel_mult) - 1
                else nn.Identity()
            )
            self.down_levels.append(
                nn.ModuleDict(
                    {
                        "block1": FlowPureResidualBlock(
                            current_channels, out_channels, time_dim, cfg.dropout
                        ),
                        "block2": FlowPureResidualBlock(
                            out_channels, out_channels, time_dim, cfg.dropout
                        ),
                        "attention": attention,
                        "downsample": downsample,
                    }
                )
            )
            skip_channels.append(out_channels)
            current_channels = out_channels
            if level_index < len(cfg.channel_mult) - 1:
                current_resolution //= 2

        self.middle1 = FlowPureResidualBlock(
            current_channels, current_channels, time_dim, cfg.dropout
        )
        self.middle_attention = FlowPureSelfAttention(current_channels)
        self.middle2 = FlowPureResidualBlock(
            current_channels, current_channels, time_dim, cfg.dropout
        )

        self.up_levels = nn.ModuleList()
        for reverse_index, (multiplier, skip_ch) in enumerate(
            zip(reversed(cfg.channel_mult), reversed(skip_channels))
        ):
            out_channels = cfg.base_channels * multiplier
            attention = (
                FlowPureSelfAttention(out_channels)
                if current_resolution in cfg.attention_resolutions
                else nn.Identity()
            )
            upsample = (
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="nearest"),
                    nn.Conv2d(out_channels, out_channels, 3, padding=1),
                )
                if reverse_index < len(cfg.channel_mult) - 1
                else nn.Identity()
            )
            self.up_levels.append(
                nn.ModuleDict(
                    {
                        "block1": FlowPureResidualBlock(
                            current_channels + skip_ch,
                            out_channels,
                            time_dim,
                            cfg.dropout,
                        ),
                        "block2": FlowPureResidualBlock(
                            out_channels, out_channels, time_dim, cfg.dropout
                        ),
                        "attention": attention,
                        "upsample": upsample,
                    }
                )
            )
            current_channels = out_channels
            if reverse_index < len(cfg.channel_mult) - 1:
                current_resolution *= 2

        self.output_norm = nn.GroupNorm(
            _flowpure_group_count(cfg.base_channels), cfg.base_channels
        )
        self.output_conv = nn.Conv2d(
            cfg.base_channels, cfg.in_channels, 3, padding=1
        )
        # 初始速度场为 0，因此未训练模型的净化操作从恒等映射开始。
        nn.init.zeros_(self.output_conv.weight)
        nn.init.zeros_(self.output_conv.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 0:
            t = t.expand(x.shape[0])
        temb = self.time_embedding(t)
        h = self.input_conv(x)
        skips: List[torch.Tensor] = []

        for level in self.down_levels:
            h = level["block1"](h, temb)
            h = level["block2"](h, temb)
            h = level["attention"](h)
            skips.append(h)
            h = level["downsample"](h)

        h = self.middle1(h, temb)
        h = self.middle_attention(h)
        h = self.middle2(h, temb)

        for level in self.up_levels:
            skip = skips.pop()
            if h.shape[-2:] != skip.shape[-2:]:
                h = F.interpolate(h, size=skip.shape[-2:], mode="nearest")
            h = torch.cat([h, skip], dim=1)
            h = level["block1"](h, temb)
            h = level["block2"](h, temb)
            h = level["attention"](h)
            h = level["upsample"](h)

        return self.output_conv(F.silu(self.output_norm(h)))


def flowpure_cfm_loss(
    model: nn.Module, x_adv: torch.Tensor, x_clean: torch.Tensor
) -> torch.Tensor:
    """Deterministic OT-CFM（sigma=0）：x_t=(1-t)x_adv+t*x_clean。"""

    batch = x_clean.shape[0]
    t = torch.rand(batch, device=x_clean.device, dtype=x_clean.dtype)
    t_image = t[:, None, None, None]
    x_t = (1.0 - t_image) * x_adv + t_image * x_clean
    target_velocity = x_clean - x_adv
    predicted_velocity = model(x_t, t)
    return F.mse_loss(predicted_velocity, target_velocity)


@torch.no_grad()
def flowpure_euler(
    model: nn.Module, x: torch.Tensor, steps: int = 10
) -> torch.Tensor:
    """从 t=0 积分到 t=1；steps 就是 FlowPure 推理 NFE。"""

    if steps <= 0:
        raise ValueError("euler_steps 必须大于 0")
    dt = 1.0 / steps
    h = x
    for index in range(steps):
        t = torch.full(
            (x.shape[0],), index * dt, device=x.device, dtype=x.dtype
        )
        h = h + dt * model(h, t)
    return h.clamp(0.0, 1.0)


# =============================================================================
# 4. FlowPure 新训练代码
# =============================================================================


def flowpure_seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _flowpure_dataset(split: str, size: int):
    """只从当前 bld.DFIRE_ROOT 构建 notebook 已定义的数据集。"""

    dataset = DFireDetDataset(FLOWPURE_DFIRE_ROOT, split=split, size=size)
    if len(dataset) == 0:
        raise FileNotFoundError(
            f"在 {FLOWPURE_DFIRE_ROOT}/{split}/images 下没有找到 JPG。"
            "请确认前面的 Kaggle 下载/解压单元已运行。"
        )
    return dataset


def _flowpure_atomic_save(payload: Dict, path: str) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, destination)


def flowpure_save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler,
    optimizer_step: int,
    cfg: FlowPureConfig,
    model_cfg: FlowPureModelConfig,
) -> None:
    _flowpure_atomic_save(
        {
            "format": "flowpure_dfire_notebook_extension_v2",
            "optimizer_step": optimizer_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "flowpure_config": asdict(cfg),
            "model_config": asdict(model_cfg),
            "dfire_root_used": FLOWPURE_DFIRE_ROOT,
        },
        path,
    )


def flowpure_load_checkpoint(
    path: str, device: torch.device
) -> Tuple[FlowPureUNet, Dict]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到 FlowPure checkpoint: {path}")
    raw = torch.load(path, map_location=device)
    if not isinstance(raw, dict) or "model_state_dict" not in raw:
        raise ValueError("该文件不是本脚本保存的 FlowPure checkpoint")

    model_cfg_dict = dict(raw.get("model_config", {}))
    for name in ("channel_mult", "attention_resolutions"):
        if name in model_cfg_dict:
            model_cfg_dict[name] = tuple(model_cfg_dict[name])
    model_cfg = FlowPureModelConfig(**model_cfg_dict)
    model = FlowPureUNet(model_cfg).to(device)
    model.load_state_dict(raw["model_state_dict"], strict=True)
    return model, raw


def train_flowpure_dfire(
    cfg: FlowPureConfig = flowpure_cfg,
    model_cfg: FlowPureModelConfig = flowpure_model_cfg,
    *,
    resume: bool = False,
    overwrite: bool = False,
) -> str:
    """
    训练 FlowPure。

    resume=False, overwrite=True  -> 从头训练并覆盖同名 checkpoint
    resume=True                   -> 从 checkpoint 的 optimizer_step 继续
    """

    device = torch.device(bld.DEVICE)
    flowpure_seed_everything(cfg.train_seed)
    flowpure_prepare_notebook_attack(cfg)

    checkpoint_exists = os.path.exists(cfg.checkpoint)
    if checkpoint_exists and not resume and not overwrite:
        raise FileExistsError(
            f"checkpoint 已存在：{cfg.checkpoint}。继续训练请传 resume=True；"
            "确认要从头覆盖才传 overwrite=True。"
        )

    dataset = _flowpure_dataset(cfg.train_split, cfg.detector_size)
    generator = torch.Generator().manual_seed(cfg.train_seed)
    loader = DataLoader(
        dataset,
        batch_size=cfg.physical_batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=det_collate,
        drop_last=True,
        generator=generator,
    )
    print(
        f"D-Fire train: root={FLOWPURE_DFIRE_ROOT}, images={len(dataset)}, "
        f"physical_batch={cfg.physical_batch_size}, "
        f"effective_batch={cfg.physical_batch_size * cfg.gradient_accumulation}"
    )

    model = FlowPureUNet(model_cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    amp_enabled = cfg.use_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)
    optimizer_step = 0

    if resume:
        if not checkpoint_exists:
            raise FileNotFoundError(
                f"resume=True，但 checkpoint 不存在：{cfg.checkpoint}"
            )
        loaded_model, raw = flowpure_load_checkpoint(cfg.checkpoint, device)
        if loaded_model.cfg != model_cfg:
            raise ValueError("当前 FlowPureModelConfig 与 checkpoint 架构不一致")
        model.load_state_dict(loaded_model.state_dict(), strict=True)
        optimizer.load_state_dict(raw["optimizer_state_dict"])
        if raw.get("scaler_state_dict"):
            scaler.load_state_dict(raw["scaler_state_dict"])
        optimizer_step = int(raw.get("optimizer_step", 0))
        print(f"已从 optimizer step {optimizer_step} 恢复：{cfg.checkpoint}")

    if optimizer_step >= cfg.max_steps:
        print(
            f"checkpoint 已到 step {optimizer_step}，不低于 max_steps={cfg.max_steps}，"
            "无需继续训练"
        )
        return cfg.checkpoint

    model.train()
    optimizer.zero_grad(set_to_none=True)
    micro_step = 0
    running_loss = 0.0
    running_batches = 0
    dropped_boxes_total = 0
    started = time.time()

    progress = tqdm(
        total=cfg.max_steps,
        initial=optimizer_step,
        desc="FlowPure-PGD optimizer steps",
    )
    while optimizer_step < cfg.max_steps:
        for x_clean_384, targets, _paths in loader:
            x_clean_384 = x_clean_384.to(device, non_blocking=True)
            targets_gpu, dropped = flowpure_sanitize_targets(targets, device)
            dropped_boxes_total += dropped

            epsilon = torch.rand(
                x_clean_384.shape[0], device=device, dtype=x_clean_384.dtype
            ) * cfg.train_eps_max
            x_adv_384 = flowpure_pgd_attack(
                x_clean_384,
                targets_gpu,
                eps=epsilon,
                alpha=cfg.pgd_alpha,
                steps=cfg.pgd_steps,
            )
            x_clean = F.interpolate(
                x_clean_384,
                size=(cfg.flow_size, cfg.flow_size),
                mode="bilinear",
                align_corners=False,
            )
            x_adv = F.interpolate(
                x_adv_384,
                size=(cfg.flow_size, cfg.flow_size),
                mode="bilinear",
                align_corners=False,
            )

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                cfm_loss = flowpure_cfm_loss(model, x_adv, x_clean)
                scaled_loss = cfm_loss / cfg.gradient_accumulation

            scaler.scale(scaled_loss).backward()
            running_loss += float(cfm_loss.detach().item())
            running_batches += 1
            micro_step += 1

            if micro_step % cfg.gradient_accumulation != 0:
                continue

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            optimizer_step += 1
            progress.update(1)

            if optimizer_step % cfg.log_every == 0:
                elapsed = max(time.time() - started, 1e-6)
                mean_loss = running_loss / max(running_batches, 1)
                progress.set_postfix(
                    cfm_loss=f"{mean_loss:.7f}",
                    steps_per_s=f"{optimizer_step / elapsed:.3f}",
                    dropped_boxes=dropped_boxes_total,
                )
                running_loss = 0.0
                running_batches = 0

            if optimizer_step % cfg.snapshot_every == 0:
                flowpure_save_checkpoint(
                    cfg.checkpoint,
                    model,
                    optimizer,
                    scaler,
                    optimizer_step,
                    cfg,
                    model_cfg,
                )
                print(f"\n已保存 FlowPure checkpoint: {cfg.checkpoint}")

            if optimizer_step >= cfg.max_steps:
                break

    progress.close()
    flowpure_save_checkpoint(
        cfg.checkpoint,
        model,
        optimizer,
        scaler,
        optimizer_step,
        cfg,
        model_cfg,
    )
    print(f"训练完成：step={optimizer_step}, checkpoint={cfg.checkpoint}")
    return cfg.checkpoint


# =============================================================================
# 5. FlowPure 新评测代码
#    复用 notebook 的 save_as_dfire_split 和 eval_map
# =============================================================================


def _flowpure_output_count(split_dir: str) -> int:
    return len(list((Path(split_dir) / "images").glob("*.jpg")))


def evaluate_flowpure_dfire(
    cfg: FlowPureConfig = flowpure_cfg,
) -> Dict[str, Tuple[float, float]]:
    """
    一次生成并检测六组图像：

      original_clean        原始干净图
      adv_original          PGD 后原图
      clean_resize_baseline 干净图 384->128->384
      adv_resize_baseline   对抗图 384->128->384
      clean_flowpure        干净图经过 FlowPure
      adv_flowpure          对抗图经过 FlowPure
    """

    device = torch.device(bld.DEVICE)
    flowpure_seed_everything(cfg.attack_seed)
    flowpure_prepare_notebook_attack(cfg)

    model, raw_checkpoint = flowpure_load_checkpoint(cfg.checkpoint, device)
    model.eval()
    if model.cfg.image_size != cfg.flow_size:
        raise ValueError(
            f"checkpoint image_size={model.cfg.image_size}，"
            f"但 cfg.flow_size={cfg.flow_size}"
        )

    dataset = _flowpure_dataset(cfg.eval_split, cfg.detector_size)
    if cfg.max_eval_images is not None:
        if cfg.max_eval_images <= 0:
            raise ValueError("max_eval_images 必须为正整数或 None")
        dataset.img_paths = dataset.img_paths[: cfg.max_eval_images]

    expected = (
        cfg.max_eval_images
        if cfg.max_eval_images is not None
        else cfg.expected_eval_images
    )
    if len(dataset) != expected:
        raise RuntimeError(
            f"{FLOWPURE_DFIRE_ROOT}/{cfg.eval_split}/images 实际发现 "
            f"{len(dataset)} 张图，但本次期望 {expected} 张。"
        )

    loader = DataLoader(
        dataset,
        batch_size=cfg.eval_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=det_collate,
    )

    checkpoint_stamp = (
        f"{os.path.getsize(cfg.checkpoint)}_{int(os.path.getmtime(cfg.checkpoint))}"
    )
    schema = "flowpure_pgd_notebook_v2_realistic"
    out_root = os.path.join(
        cfg.output_root,
        f"dfire_adv_eval_flowpure_{checkpoint_stamp}"
        f"_eps{cfg.eval_eps:.6f}_seed{cfg.attack_seed}"
        f"_batch{cfg.eval_batch_size}_flow{cfg.flow_size}"
        f"_nfe{cfg.euler_steps}_{schema}",
    )
    output_splits = (
        "original_clean",
        "adv_original",
        "clean_resize_baseline",
        "adv_resize_baseline",
        "clean_flowpure",
        "adv_flowpure",
    )
    for split in output_splits:
        Path(out_root, split, "images").mkdir(parents=True, exist_ok=True)
        Path(out_root, split, "labels").mkdir(parents=True, exist_ok=True)

    expected_names = {Path(path).name for path in dataset.img_paths}
    path_by_name = {Path(path).name: path for path in dataset.img_paths}
    if len(path_by_name) != len(dataset):
        raise RuntimeError("test 集出现重复文件名，无法安全断点续跑")

    progress_path = Path(out_root) / "done.txt"
    done_raw = (
        set(progress_path.read_text(encoding="utf-8").split())
        if progress_path.exists()
        else set()
    )
    done = {
        name
        for name in done_raw
        if name in expected_names
        and all(
            Path(out_root, split, "images", Path(path_by_name[name]).stem + ".jpg")
            .exists()
            for split in output_splits
        )
    }
    print(f"FlowPure 评测断点：{len(done)}/{expected}；输出目录：{out_root}")

    with open(progress_path, "a", encoding="utf-8") as progress_handle:
        for x_clean_384, targets, paths in tqdm(
            loader, desc="D-Fire PGD -> FlowPure"
        ):
            names = [Path(path).name for path in paths]
            if all(name in done for name in names):
                continue

            x_clean_384 = x_clean_384.to(device, non_blocking=True)
            targets_gpu, _ = flowpure_sanitize_targets(targets, device)

            batch_seed = cfg.attack_seed + zlib.crc32(
                "|".join(names).encode("utf-8")
            )
            torch.manual_seed(batch_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(batch_seed)

            x_adv_384 = flowpure_pgd_attack(
                x_clean_384,
                targets_gpu,
                eps=cfg.eval_eps,
                alpha=cfg.pgd_alpha,
                steps=cfg.pgd_steps,
            )
            x_clean_128 = F.interpolate(
                x_clean_384,
                size=(cfg.flow_size, cfg.flow_size),
                mode="bilinear",
                align_corners=False,
            )
            x_adv_128 = F.interpolate(
                x_adv_384,
                size=(cfg.flow_size, cfg.flow_size),
                mode="bilinear",
                align_corners=False,
            )

            with torch.no_grad(), torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=cfg.use_amp and device.type == "cuda",
            ):
                x_clean_flow_128 = flowpure_euler(
                    model, x_clean_128, cfg.euler_steps
                )
                x_adv_flow_128 = flowpure_euler(
                    model, x_adv_128, cfg.euler_steps
                )

            resize_kwargs = dict(
                size=(cfg.detector_size, cfg.detector_size),
                mode="bilinear",
                align_corners=False,
            )
            batches = {
                "original_clean": x_clean_384,
                "adv_original": x_adv_384,
                "clean_resize_baseline": F.interpolate(
                    x_clean_128, **resize_kwargs
                ),
                "adv_resize_baseline": F.interpolate(x_adv_128, **resize_kwargs),
                "clean_flowpure": F.interpolate(
                    x_clean_flow_128.float(), **resize_kwargs
                ),
                "adv_flowpure": F.interpolate(
                    x_adv_flow_128.float(), **resize_kwargs
                ),
            }
            for split, images in batches.items():
                save_as_dfire_split(images, paths, os.path.join(out_root, split))

            for name in names:
                progress_handle.write(name + "\n")
                done.add(name)
            progress_handle.flush()

    missing = expected_names - done
    if missing:
        raise RuntimeError(f"仍有 {len(missing)} 张测试图没有完成")
    for split in output_splits:
        count = _flowpure_output_count(os.path.join(out_root, split))
        print(f"{split:24s} images={count}/{expected}")
        if count != expected:
            raise RuntimeError(
                f"{split} 中有 {count} 张图，预期 {expected}；停止 mAP"
            )

    results: Dict[str, Tuple[float, float]] = {}
    print("\n=== FlowPure-PGD D-Fire realistic / preprocessor-blind ===")
    for split in output_splits:
        # 使用 notebook 原来的 eval_map(save_dir, img_size=384)。
        map50, map5095 = eval_map(
            os.path.join(out_root, split), img_size=cfg.detector_size
        )
        results[split] = (float(map50), float(map5095))
        print(f"{split:24s} mAP50={map50:.4f}  mAP50-95={map5095:.4f}")

    clean50, clean5095 = results["original_clean"]
    adv50, adv5095 = results["adv_original"]
    flow50, flow5095 = results["adv_flowpure"]
    resize50, resize5095 = results["adv_resize_baseline"]
    recovery50 = (flow50 - adv50) / max(clean50 - adv50, 1e-8)
    recovery5095 = (flow5095 - adv5095) / max(clean5095 - adv5095, 1e-8)
    resize_recovery50 = (resize50 - adv50) / max(clean50 - adv50, 1e-8)
    clean_drop50 = clean50 - results["clean_flowpure"][0]
    flow_over_resize50 = flow50 - resize50

    print(f"\nFlowPure mAP50 恢复率 = {recovery50:.1%}")
    print(f"FlowPure mAP50-95 恢复率 = {recovery5095:.1%}")
    print(f"仅 384->128->384 的 mAP50 恢复率 = {resize_recovery50:.1%}")
    print(f"干净图经过 FlowPure 的 mAP50 损失 = {clean_drop50:+.4f}")
    print(f"FlowPure 相对仅 resize 的 mAP50 额外增益 = {flow_over_resize50:+.4f}")

    report = {
        "schema": schema,
        "dfire_root_used": FLOWPURE_DFIRE_ROOT,
        "checkpoint": cfg.checkpoint,
        "checkpoint_optimizer_step": int(
            raw_checkpoint.get("optimizer_step", -1)
        ),
        "flowpure_config": asdict(cfg),
        "model_config": asdict(model.cfg),
        "results": {
            key: {"map50": value[0], "map50_95": value[1]}
            for key, value in results.items()
        },
        "derived": {
            "flowpure_recovery_map50": recovery50,
            "flowpure_recovery_map50_95": recovery5095,
            "resize_only_recovery_map50": resize_recovery50,
            "clean_flowpure_drop_map50": clean_drop50,
            "flowpure_gain_over_resize_map50": flow_over_resize50,
        },
    }
    report_path = Path(out_root) / "metrics.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"指标已保存：{report_path}")
    return results


# =============================================================================
# 6. 本文件只定义新功能，不会在载入时自动训练
# =============================================================================


print("FlowPure 新代码已载入，但尚未开始训练或评测。")
print("先检查：len(_flowpure_dataset('train', 384)), len(_flowpure_dataset('test', 384))")
print("冒烟训练建议单独使用 smoke checkpoint，并只跑 2 个 optimizer steps")
print("继续训练：调大 flowpure_cfg.max_steps；然后调用 train_flowpure_dfire(resume=True)")
print("完整评测：flowpure_cfg.max_eval_images=None；然后调用 evaluate_flowpure_dfire()")

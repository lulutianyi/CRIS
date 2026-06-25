"""
第二步：三方检测精度对比
原图 vs task-aware重建图(bpp≈0.42) vs low-bpp重建图(bpp≈0.1)

流程：
  1. 加载训练好的检测器（train_detector.py 产出的 best.pt）
  2. 分别加载两个压缩模型权重
  3. 对 D-Fire test 集中的每张图：
       - 用压缩模型生成重建图
       - 把重建图写到磁盘，构造一套临时的"重建图数据集"（保持原标注不变，
         因为压缩不改变物体位置，只改变像素内容）
  4. 用同一个检测器分别在三套图像上跑 model.val()，对比 mAP50 / mAP50-95
"""

import os
import shutil
import glob

import torch
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import save_image

from compressai.models import Cheng2020Anchor
from ultralytics import YOLO

# 从纯模型定义文件导入，不会触发训练（不要从 train_residual_enhance.py 导入）
from residual_unet_model import ResidualUNet

# =====================================================================
# 配置
# =====================================================================
DATA_ROOT     = "/content/D-Fire"
IMG_SIZE      = 384
BATCH_SIZE    = 16
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DETECTOR_CKPT       = "/content/yolov8n_dfire_detector.pt"
TASK_AWARE_CKPT     = "/content/cheng2020_task_aware_best.pth"   # bpp≈0.42
LOW_BPP_CKPT        = "/content/cheng2020_low_bpp_best.pth"      # bpp≈0.1（纯压缩，不增强）
RESIDUAL_CKPT       = "outputs_residual_enhance_low_bpp/best_model.pth"  # 残差增强网络

# 重建图临时存放位置（每套图像一个文件夹，标注直接软链接复用原 labels）
RECON_ROOT = "/content/recon_eval"


# =====================================================================
# 数据集：只需要原图路径列表，不需要标注（标注会原样复用）
# =====================================================================
class ImagePathDataset(Dataset):
    def __init__(self, root, split="test"):
        img_dir = os.path.join(root, split, "images")
        self.paths = sorted(
            glob.glob(os.path.join(img_dir, "*.jpg"))
            + glob.glob(os.path.join(img_dir, "*.png"))
        )
        self.transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        stem = os.path.splitext(os.path.basename(path))[0]
        return img, stem


def build_recon_dataset(model, loader, out_name):
    """
    用给定压缩模型对所有图片做重建，保存到 RECON_ROOT/{out_name}/images，
    并把对应的 labels 文件夹软链接过去（重建不改变物体位置，标注无需重新生成）。
    """
    img_out_dir = os.path.join(RECON_ROOT, out_name, "images")
    lbl_out_dir = os.path.join(RECON_ROOT, out_name, "labels")
    os.makedirs(img_out_dir, exist_ok=True)

    # labels 直接软链接原始标注目录，省时间省空间
    src_lbl_dir = os.path.join(DATA_ROOT, "test", "labels")
    if os.path.exists(lbl_out_dir) or os.path.islink(lbl_out_dir):
        os.remove(lbl_out_dir) if os.path.islink(lbl_out_dir) else shutil.rmtree(lbl_out_dir)
    os.symlink(src_lbl_dir, lbl_out_dir)

    model.eval()
    with torch.no_grad():
        for imgs, stems in loader:
            imgs = imgs.to(DEVICE)
            out = model(imgs)
            x_hat = out["x_hat"].clamp(0, 1)
            for i, stem in enumerate(stems):
                save_image(x_hat[i].cpu(), os.path.join(img_out_dir, f"{stem}.jpg"))

    print(f"已生成重建图数据集: {img_out_dir}  ({len(os.listdir(img_out_dir))} 张)")
    return img_out_dir


def build_recon_enhanced_dataset(compress_model, residual_model, loader, out_name):
    """
    [新增] 两步流程：先用压缩模型生成 recon，再用残差增强网络修正，
    enhanced = recon + residual_model(recon)
    这是评估"压缩+残差增强"实际效果的关键函数，区别于 build_recon_dataset
    （只做压缩，不做后续增强）。
    """
    img_out_dir = os.path.join(RECON_ROOT, out_name, "images")
    lbl_out_dir = os.path.join(RECON_ROOT, out_name, "labels")
    os.makedirs(img_out_dir, exist_ok=True)

    src_lbl_dir = os.path.join(DATA_ROOT, "test", "labels")
    if os.path.exists(lbl_out_dir) or os.path.islink(lbl_out_dir):
        os.remove(lbl_out_dir) if os.path.islink(lbl_out_dir) else shutil.rmtree(lbl_out_dir)
    os.symlink(src_lbl_dir, lbl_out_dir)

    compress_model.eval()
    residual_model.eval()
    with torch.no_grad():
        for imgs, stems in loader:
            imgs = imgs.to(DEVICE)
            out = compress_model(imgs)
            recon = out["x_hat"].clamp(0, 1)

            pred_residual = residual_model(recon)
            enhanced = (recon + pred_residual).clamp(0, 1)

            for i, stem in enumerate(stems):
                save_image(enhanced[i].cpu(), os.path.join(img_out_dir, f"{stem}.jpg"))

    print(f"已生成[压缩+残差增强]数据集: {img_out_dir}  ({len(os.listdir(img_out_dir))} 张)")
    return img_out_dir


def make_eval_yaml(img_dir_name):
    """为某一套图像（原图/重建图）生成一个独立的 dfire yaml 给 YOLO val 用。"""
    import yaml
    yaml_path = os.path.join(RECON_ROOT, f"{img_dir_name}.yaml")
    cfg = {
        "path":  os.path.join(RECON_ROOT, img_dir_name) if img_dir_name != "original" else DATA_ROOT,
        "train": "images",   # val 模式不会用到，占位
        "val":   "images" if img_dir_name != "original" else "test/images",
        "names": {0: "fire", 1: "smoke"},
    }
    with open(yaml_path, "w") as f:
        yaml.dump(cfg, f)
    return yaml_path


# =====================================================================
# 主流程
# =====================================================================
print(f"Device: {DEVICE}", flush=True)
os.makedirs(RECON_ROOT, exist_ok=True)

# ---- 1. 加载检测器（固定不变，三套图像用同一个检测器评估）----
print("加载检测器...", flush=True)
detector = YOLO(DETECTOR_CKPT)

# ---- 2. 准备测试集图片路径 ----
dataset = ImagePathDataset(DATA_ROOT, split="test")
loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
print(f"测试集图片数: {len(dataset)}", flush=True)

results_summary = {}

# ---- 3a. 原图：直接用官方 test 目录评估，不需要重建 ----
print("\n===== [1/3] 原图 baseline =====", flush=True)
yaml_original = make_eval_yaml("original")
metrics_orig = detector.val(data=yaml_original, imgsz=IMG_SIZE, split="val")
results_summary["original"] = {
    "mAP50":    metrics_orig.box.map50,
    "mAP50-95": metrics_orig.box.map,
}

# ---- 3b. task-aware 重建图 (bpp≈0.42) ----
print("\n===== [2/3] task-aware 重建图 (bpp≈0.42) =====", flush=True)
model_ta = Cheng2020Anchor(N=128).to(DEVICE)
ckpt_ta = torch.load(TASK_AWARE_CKPT, map_location=DEVICE)
model_ta.load_state_dict(ckpt_ta["model"])
print(f"已加载 task-aware 权重 (PSNR={ckpt_ta['val_psnr']:.2f}, bpp={ckpt_ta.get('val_bpp', float('nan')):.4f})")

build_recon_dataset(model_ta, loader, "task_aware")
yaml_ta = make_eval_yaml("task_aware")
metrics_ta = detector.val(data=yaml_ta, imgsz=IMG_SIZE, split="val")
results_summary["task_aware"] = {
    "mAP50":    metrics_ta.box.map50,
    "mAP50-95": metrics_ta.box.map,
}

del model_ta
torch.cuda.empty_cache()

# ---- 3c. low-bpp 重建图 (bpp≈0.1)，纯压缩不增强 ----
print("\n===== [3/4] low-bpp 重建图 (bpp≈0.1，未增强) =====", flush=True)
model_lb = Cheng2020Anchor(N=128).to(DEVICE)
ckpt_lb = torch.load(LOW_BPP_CKPT, map_location=DEVICE)
model_lb.load_state_dict(ckpt_lb["model"])
print(f"已加载 low-bpp 权重 (PSNR={ckpt_lb['val_psnr']:.2f}, bpp={ckpt_lb.get('val_bpp', float('nan')):.4f})")

build_recon_dataset(model_lb, loader, "low_bpp")
yaml_lb = make_eval_yaml("low_bpp")
metrics_lb = detector.val(data=yaml_lb, imgsz=IMG_SIZE, split="val")
results_summary["low_bpp"] = {
    "mAP50":    metrics_lb.box.map50,
    "mAP50-95": metrics_lb.box.map,
}

# ---- 3d. [新增] low-bpp + 残差增强 ----
print("\n===== [4/4] low-bpp 重建图 + 残差增强 =====", flush=True)
residual_model = ResidualUNet(base_ch=32).to(DEVICE)
ckpt_res = torch.load(RESIDUAL_CKPT, map_location=DEVICE)
residual_model.load_state_dict(ckpt_res["model"])
print(f"已加载残差增强权重 (Val PSNR={ckpt_res['val_psnr']:.2f}, "
      f"baseline PSNR={ckpt_res['val_psnr_baseline']:.2f}, "
      f"target={ckpt_res['target']})")

build_recon_enhanced_dataset(model_lb, residual_model, loader, "low_bpp_enhanced")
yaml_enhanced = make_eval_yaml("low_bpp_enhanced")
metrics_enhanced = detector.val(data=yaml_enhanced, imgsz=IMG_SIZE, split="val")
results_summary["low_bpp_enhanced"] = {
    "mAP50":    metrics_enhanced.box.map50,
    "mAP50-95": metrics_enhanced.box.map,
}

del model_lb, residual_model
torch.cuda.empty_cache()

# =====================================================================
# 汇总结果
# =====================================================================
print("\n" + "=" * 60)
print("四方检测精度对比汇总")
print("=" * 60)
print(f"{'数据集':<24}{'mAP50':>10}{'mAP50-95':>12}")
for name, m in results_summary.items():
    print(f"{name:<24}{m['mAP50']:>10.4f}{m['mAP50-95']:>12.4f}")

orig_map50 = results_summary["original"]["mAP50"]
ta_drop  = (orig_map50 - results_summary["task_aware"]["mAP50"]) / orig_map50 * 100
lb_drop  = (orig_map50 - results_summary["low_bpp"]["mAP50"]) / orig_map50 * 100
enh_drop = (orig_map50 - results_summary["low_bpp_enhanced"]["mAP50"]) / orig_map50 * 100

print(f"\ntask-aware(bpp=0.42)        相对原图 mAP50 下降: {ta_drop:.1f}%")
print(f"low-bpp(bpp=0.1, 未增强)    相对原图 mAP50 下降: {lb_drop:.1f}%")
print(f"low-bpp + 残差增强          相对原图 mAP50 下降: {enh_drop:.1f}%")

recovery = lb_drop - enh_drop
print(f"\n残差增强挽回的 mAP50 损失: {recovery:+.1f} 个百分点")
if recovery > 5:
    print("→ 残差增强有明显效果，值得继续优化这条路线")
elif recovery > 0:
    print("→ 残差增强有一定效果，但提升有限，可能需要增大网络容量或调整损失函数")
else:
    print("→ 残差增强没有带来检测精度提升，PSNR的提升未必转化为任务收益，需要重新设计训练目标")

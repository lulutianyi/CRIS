# HF-AE 公共数据集子集评测

脚本：`experiments/public_subset_eval.py`

该脚本不复制 `gcn.ipynb` 的网络定义。先在 Colab 中运行 notebook 至 HF-AE/GCN 模型定义完成，并保持所有训练开关为 `False`，然后通过 `%run -i` 复用当前 notebook 环境。

## 评测含义

- CIFAR-10：报告 128×128 工作分辨率和原生 32×32 下的 PSNR、SSIM、MAE、高通 PSNR。D-Fire 的火焰/烟雾检测器不能用于 CIFAR 分类评价。
- COCO：报告 128×128 重建指标。添加 `--coco-detection` 后，使用 COCO 预训练 YOLO 分别评价原图与重建图，并报告 mAP 保留率。
- 两个数据集都只做推理，不训练 HF-AE。
- 子集由 `--seed` 固定，比较 Learned-v2 和 HF-AE 时必须使用相同 subset size 与 seed。

## CIFAR-10

首次运行允许 torchvision 下载数据：

```python
%run -i "/content/drive/MyDrive/CRIS/bld & purifier/experiments/public_subset_eval.py" \
  --dataset cifar10 \
  --subset-size 1000 \
  --seed 20261010 \
  --download \
  --checkpoint "/content/drive/MyDrive/gcn_ae_checkpoints/skip_learned_v2_hf/gcn_skip_learned_v2_hf_taskaware_best.pt" \
  --model-name hf_ae \
  --output-dir /content/public_subset_eval
```

比较旧 Learned-v2 时仅替换权重和名字：

```python
%run -i "/content/drive/MyDrive/CRIS/bld & purifier/experiments/public_subset_eval.py" \
  --dataset cifar10 \
  --subset-size 1000 \
  --seed 20261010 \
  --download \
  --checkpoint "/content/drive/MyDrive/gcn_ae_checkpoints/skip_learned_v2/gcn_skip_learned_v2_taskaware_best.pt" \
  --model-name learned_v2 \
  --output-dir /content/public_subset_eval
```

## COCO 2017

脚本预期以下标准目录：

```text
/content/coco/val2017/
/content/coco/annotations/instances_val2017.json
```

重建指标与 COCO 检测 mAP：

```python
%run -i "/content/drive/MyDrive/CRIS/bld & purifier/experiments/public_subset_eval.py" \
  --dataset coco \
  --subset-size 500 \
  --seed 20261010 \
  --coco-images /content/coco/val2017 \
  --coco-annotations /content/coco/annotations/instances_val2017.json \
  --coco-detection \
  --coco-yolo yolov8n.pt \
  --coco-det-batch 4 \
  --checkpoint "/content/drive/MyDrive/gcn_ae_checkpoints/skip_learned_v2_hf/gcn_skip_learned_v2_hf_taskaware_best.pt" \
  --model-name hf_ae \
  --output-dir /content/public_subset_eval
```

需要安装 COCO API：

```python
!pip -q install pycocotools
```

## 推荐规模

- 首次冒烟：CIFAR-10 `100`，COCO `50`。
- 正式子集：CIFAR-10 `1000`（每类100张），COCO `500`。
- Learned-v2 与 HF-AE 必须使用同一个 `--seed 20261010`。

结果保存在：

```text
<output-dir>/<model-name>_<dataset>_n<subset-size>_seed<seed>/summary.json
```

COCO 检测结果应重点报告原图 mAP、重建图 mAP 和二者比值。由于模型在 D-Fire 上训练，公共数据集结果属于跨域泛化测试，不应与 D-Fire PGD 恢复率混为同一个指标。

## COCO 128分辨率对照

为了区分“128×128缩放损失”和“AE压缩/跨域损失”，使用独立脚本：

```python
%run "/content/drive/MyDrive/CRIS/bld & purifier/experiments/coco_resize_only_eval.py" \
  --coco-images /content/coco/val2017 \
  --coco-annotations /content/coco/annotations/instances_val2017.json \
  --subset-size 500 \
  --seed 20261010 \
  --ae-size 128 \
  --detector yolov8n.pt \
  --det-batch 2 \
  --output-dir /content/coco_resize_only_eval
```

该脚本不加载 `gcn.ipynb` 或HF-AE，只执行“原图→128×128→原尺寸”，并报告
原图mAP、resize-only mAP和保留率。必须与HF-AE评测使用相同的subset size和seed。

在 Tesla T4 上，COCO 检测默认使用 `--coco-det-batch 4`，并在检测前把
HF-AE 移到 CPU、释放 CUDA 缓存。如果 notebook 中仍保留了多个模型导致显存不足，
将该参数进一步改成 `--coco-det-batch 2` 或 `1`。

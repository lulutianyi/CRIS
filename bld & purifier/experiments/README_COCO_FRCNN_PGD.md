# COCO + Faster R-CNN PGD：前三步

把以下三个文件放到 Colab 的同一目录（示例为 `/content`）：

- `coco_frcnn_common.py`
- `eval_coco_frcnn.py`
- `generate_coco_frcnn_pgd.py`

数据路径：

```text
/content/coco/val2017
/content/coco/annotations/instances_val2017.json
```

## 1. 先测 20 张 clean 基线

```python
%run "/content/eval_coco_frcnn.py" \
  --images "/content/coco/val2017" \
  --annotations "/content/coco/annotations/instances_val2017.json" \
  --subset-size 20 \
  --seed 20261010 \
  --source clean \
  --output-dir "/content/drive/MyDrive/coco_frcnn_purifier/smoke_n20/clean_before_attack"
```

## 2. 在同一批图片上生成 PGD-5

```python
%run "/content/generate_coco_frcnn_pgd.py" \
  --images "/content/coco/val2017" \
  --annotations "/content/coco/annotations/instances_val2017.json" \
  --subset-size 20 \
  --seed 20261010 \
  --epsilon 0.031372549 \
  --alpha 0.007843137 \
  --steps 5 \
  --output-dir "/content/drive/MyDrive/coco_frcnn_purifier/smoke_n20/pgd5_eps8" \
  --resume
```

## 3. 用同一个 manifest 测 clean 和 adversarial mAP

```python
%run "/content/eval_coco_frcnn.py" \
  --images "/content/coco/val2017" \
  --annotations "/content/coco/annotations/instances_val2017.json" \
  --manifest "/content/drive/MyDrive/coco_frcnn_purifier/smoke_n20/pgd5_eps8/manifest.json" \
  --source clean \
  --output-dir "/content/drive/MyDrive/coco_frcnn_purifier/smoke_n20/eval_clean"

%run "/content/eval_coco_frcnn.py" \
  --images "/content/coco/val2017" \
  --annotations "/content/coco/annotations/instances_val2017.json" \
  --manifest "/content/drive/MyDrive/coco_frcnn_purifier/smoke_n20/pgd5_eps8/manifest.json" \
  --source adversarial \
  --output-dir "/content/drive/MyDrive/coco_frcnn_purifier/smoke_n20/eval_adversarial"
```

主要查看两个 summary 中的：

```text
metrics.map50_95
metrics.map50
```

只有 adversarial mAP 明显低于 clean mAP，才扩大样本数量并开始训练净化器。

## 4. 生成首批 1000 对 train2017 数据

不要用 `val2017` 训练净化器。以下命令只下载固定的 1000 张
`train2017` 图片，不需要下载完整训练集：

```python
%run "/content/prepare_coco_train_subset.py" \
  --annotations "/content/coco/annotations/instances_train2017.json" \
  --output-dir "/content/drive/MyDrive/coco_frcnn_purifier/data/train2017_clean_n1000_seed20261010" \
  --subset-size 1000 \
  --seed 20261010 \
  --workers 8 \
  --insecure
```

`--insecure` 仅用于当前 Colab 环境已经出现的 COCO 官方图片域名证书错误。
下载完成后生成 PGD-5 配对：

```python
%run "/content/generate_coco_frcnn_pgd.py" \
  --images "/content/drive/MyDrive/coco_frcnn_purifier/data/train2017_clean_n1000_seed20261010" \
  --annotations "/content/coco/annotations/instances_train2017.json" \
  --subset-size 1000 \
  --seed 20261010 \
  --epsilon 0.031372549 \
  --alpha 0.007843137 \
  --steps 5 \
  --output-dir "/content/drive/MyDrive/coco_frcnn_purifier/data/train2017_frcnn_pgd5_eps8_n1000_seed20261010" \
  --resume
```

中断后使用完全相同的命令重新运行，脚本会根据 `manifest.json` 跳过已经完成的样本。

## 5. 第一阶段净化器训练

这一阶段只需要挂载 Drive，不需要重新下载 COCO 或 Faster R-CNN。将
`train_coco_residual_purifier_stage1.py` 放到 `/content` 后运行：

```python
%run "/content/train_coco_residual_purifier_stage1.py" \
  --manifest "/content/drive/MyDrive/coco_frcnn_purifier/data/train2017_frcnn_pgd5_eps8_n1000_seed20261010/manifest.json" \
  --output-dir "/content/drive/MyDrive/coco_frcnn_purifier/checkpoints/native_residual_stage1_n1000_seed20261010" \
  --val-size 100 \
  --seed 20261010 \
  --epochs 25 \
  --batch-size 8 \
  --patch-size 256 \
  --workers 2 \
  --resume
```

每轮都会向 Drive 写入 `purifier_stage1_last.pt`、最优的
`purifier_stage1_best.pt`、`history.json` 和 PNG 格式的可视化预览。若 T4
显存不足，将 `--batch-size 8` 改为 `--batch-size 4`；若 Colab 出现
DataLoader 子进程错误，将 `--workers 2` 改为 `--workers 0`。


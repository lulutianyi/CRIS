# Experiments

当前主线是 COCO/Faster R-CNN 原生分辨率残差净化器。旧的 HF-AE/CIFAR
脚本用于保存历史实验与跨域对照，不属于当前训练链。

## 当前 COCO 主线

- `coco_frcnn_common.py`：COCO、Faster R-CNN 和评测公共函数。
- `prepare_coco_train_subset.py`：按固定 `split.json` 恢复训练图片。
- `make_coco_eval_ids.py`：生成固定且可排除旧样本的评测编号。
- `train_coco_residual_purifier_stage1.py`：定义净化器并完成配对重建训练。
- `train_coco_residual_purifier_stage2.py`：加入检测任务损失。
- `train_coco_residual_purifier_stage3.py`：自适应 PGD 训练及公共训练函数。
- `train_coco_residual_purifier_stage3_mix.py`：自适应/非自适应混合训练；当前
  Stage3-Mix 与 Stage4 都由它运行。
- `eval_coco_residual_purifier.py`：统一的非自适应和端到端自适应 PGD 评测。

统一评测脚本通过 `--attack-mode` 切换模式：

```bash
python eval_coco_residual_purifier.py \
  --attack-mode nonadaptive \
  --images /content/coco/val2017 \
  --annotations /content/coco/annotations/instances_val2017.json \
  --checkpoint /path/to/purifier.pt \
  --image-ids-json /path/to/fixed_ids.json \
  --output-dir /path/to/nonadaptive_results
```

端到端白盒评测仅需将模式改为：

```bash
--attack-mode adaptive
```

不传 `--attack-mode` 时默认保持旧行为，运行非自适应 PGD。

## 早期 COCO 数据准备

- `eval_coco_frcnn.py`：检查原始 Faster R-CNN 或替换图片目录的检测性能。
- `generate_coco_frcnn_pgd.py`：生成 Stage1 使用的固定 PGD 配对数据。

这两个脚本当前不用于在线 Stage3/Stage4 训练，但仍是 Stage1 可复现链的一部分。

## 历史 HF-AE 与跨域实验

- `public_subset_eval.py`：旧 HF-AE 的 CIFAR-10/COCO 重建评测。
- `coco_resize_only_eval.py`：128×128 resize-only COCO 对照。
- `cifar10_classification_eval.py`：CIFAR-10 分类保持率。
- `cifar10_robustness_eval.py`：CIFAR-10 扰动、PGD 与编码稳定性。
- `cifar10_hfae_adaptive_finetune.py`：旧 HF-AE CIFAR 自适应微调。

这些脚本彼此承担不同实验，不是当前 COCO 脚本的重复实现。若论文不再需要
HF-AE/CIFAR 消融，可整体移动到单独的归档目录，但不建议直接删除。

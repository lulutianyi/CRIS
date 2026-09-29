# Robust-v3 P0/P1封存与诊断脚本

入口：`core/gcn_p0_p1_audit.py`。模型定义在
`core/gcn_p0_p1_models.py`。两份脚本均为推理/评测代码，不创建优化器，
不重新训练，也不读取FlowPure或FreqPure权重。

## 运行前

1. 在Colab挂载Google Drive，并保证仓库文件可从`/content`下访问；如果仓库
   位于其他目录，所有命令都传正确的`--repo-root`。
2. 确认三份权重、D-Fire和历史共享PGD目录存在。
3. 最好先把**当时Colab实际执行的定义单元**另存为文件，放入仓库后再运行
   `archive`。严格加载只能证明state_dict名称和形状兼容，不能证明Python forward
   语义与历史运行完全相同。
4. 第一次建议加`--no-save-detections`做小范围流程验证；正式结果不要加该选项。

默认权重路径已经写入脚本：

```text
/content/drive/MyDrive/gcn_ae_checkpoints/skip_learned_v2_robust/gcn_skip_learned_v2_robust_taskaware_final.pt
/content/drive/MyDrive/gcn_ae_checkpoints/skip6144_from_robustv3_only64/seed20260921_8x500/epoch8.pt
/content/drive/MyDrive/dfire_checkpoints/yolov8n_dfire_detector.pt
```

## 推荐顺序

以下假设脚本位于`/content/core/`。

### 1. P0资料封存

```python
%run /content/core/gcn_p0_p1_audit.py \
  --stage archive \
  --repo-root /content \
  --output-root /content/drive/MyDrive/gcn_p0_p1_audit
```

该阶段计算三份权重的完整SHA256，复制并哈希模型/评测定义，保存Git提交、
未提交差异、Python和包版本，并对共享PGD的每张clean/adv PNG及标签计算SHA256。
它还保存完整D-Fire train/test文件清单。此阶段磁盘读取量较大，但可以重跑。

### 2. 固定未来训练/validation名单

```python
%run /content/core/gcn_p0_p1_audit.py \
  --stage make-splits \
  --output-root /content/drive/MyDrive/gcn_p0_p1_audit \
  --validation-count 512 \
  --split-seed 20260928
```

此操作只生成名单，不移动数据。输出明确记录：validation可能被源v3训练见过；
历史4302张测试集也不是未使用过的独立测试集。

### 3. 为固定validation生成一次共享PGD

必须使用一个全新的目录：

```python
%run /content/core/gcn_p0_p1_audit.py \
  --stage generate-val-pgd \
  --p1-shared-root /content/drive/MyDrive/dfire_fixed_val_pgd_eps8_seed20260822 \
  --output-root /content/drive/MyDrive/gcn_p0_p1_audit \
  --batch-size 8
```

PGD默认参数为ε=8/255、步长2/255、10步、逐图确定性随机初始化、seed
20260822、Ultralytics `v8DetectionLoss`，输入为RGB 384×384 `[0,1]`。
生成过程可恢复；若已有manifest与当前参数不一致，脚本会拒绝混用。

### 4. P0全量基线与扩容等价性

```python
%run /content/core/gcn_p0_p1_audit.py \
  --stage p0 \
  --output-root /content/drive/MyDrive/gcn_p0_p1_audit \
  --batch-size 2 \
  --map-batch-size 16
```

P0会严格加载原v3和扩容epoch8，并检查：

- 所有未扩宽参数逐bit相等；
- 扩宽张量的旧切片逐bit相等；
- 每张clean/adv图的main码和两路旧skip切片相等；
- v3与扩容`old_only`重建的最大绝对误差；
- 原图、攻击图、v3、扩容old_only和扩容full的mAP50/mAP50-95及恢复率。

`old_only`的唯一实现是：扩容模型正常编码，然后在`decode_from_codes`之前把
skip64通道`[2:]`和skip32通道`[8:]`置零。不会把不同长度的整条skip码直接比较。

### 5. P1固定validation码流诊断

```python
%run /content/core/gcn_p0_p1_audit.py \
  --stage p1 \
  --p1-shared-root /content/drive/MyDrive/dfire_fixed_val_pgd_eps8_seed20260822 \
  --output-root /content/drive/MyDrive/gcn_p0_p1_audit \
  --batch-size 2 \
  --map-batch-size 16
```

P1保存三路码流的逐图翻转率、概率margin和one-rate，并评测：

- clean/adv硬码；
- clean-main + adv-skips、adv-main + clean-skips两个oracle条件；
- main/skip的软码、硬码组合。

软码不会在decoder内部再次阈值化。输出JSON明确标记软码是给只用硬码训练的
decoder带来的OOD诊断，oracle也不可作为可部署净化结果；这些结果不能相加成
0.1215 mAP差值的独立模块分解。

若确实需要保存每张图的完整概率和硬码张量，加`--save-code-tensors`。数据量会
明显增大。

## 逐图检测规则

正式运行默认额外保存每张图的预测框、类别、置信度、真实框以及匹配结果。
诊断匹配固定为：confidence≥0.25、同类别、IoU≥0.50，预测按置信度降序贪心
匹配未使用GT。背景图分为`true_negative`和`background_false_positive`。
目标大小按384输入上的COCO阈值分为small（<32²）、medium（<96²）和large。

这个逐图规则只用于诊断；汇总mAP仍由Ultralytics验证器计算，参数完整写入JSON。

## 主要输出

```text
gcn_p0_p1_audit/
  archive/
    checkpoint_sha256.json
    source_snapshot/
    source_sha256.json
    git_state.json
    environment.json
    p0_shared/shared_file_hashes.jsonl
    dfire_train_inventory.jsonl
    dfire_test_inventory.jsonl
  splits/fixed_splits.json
  p0/
    inherited_state_equivalence.json
    per_image_equivalence.jsonl
    equivalence_summary.json
    metrics_p0.json
    detections/*.jsonl
  p1/
    per_image_bit_statistics.jsonl
    metrics_p1.json
    detections/*.jsonl
```

## 防误用保护

- P1默认拒绝使用与P0相同的历史测试PGD目录。
- P0默认要求4306张、4302张有效、4张排除；数量不符立即停止。
- 所有模型权重严格加载；架构或state_dict不匹配立即停止。
- 扩容继承参数、旧码切片不等价时，P0在计算mAP前停止。
- 输出图片统一保存为384×384 RGB PNG，避免JPEG或隐式检测器缩放改变口径。

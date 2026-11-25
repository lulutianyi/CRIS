# DiffPure / FlowPure × ZeroPur ResNet18

这组脚本固定使用 ZeroPur 官方 `PreActResNet18 (all)`，并在同一批缓存样本上报告：

- 净化后的 clean accuracy；
- 固定、非自适应 AutoAttack 样本的净化后 accuracy；
- 端到端 BPDA-PGD-20 的净化后 accuracy。

攻击半径均为 `8/255`，BPDA 步长默认为 `2/255`。

## 需要的权重

```text
/content/FlowPure/resources/checkpoints/score_sde/checkpoint_c10.pth
/content/FlowPure/resources/checkpoints/flowpure_pgd/checkpoint_c10.pth
/content/zeropur_workspace/official/zero_pur/checkpoint/resnet18_cifar10_all/ckpt.pth
```

第一项用于 DiffPure。第二项用于默认的 FlowPure（`pgd_flowpure`）；官方仓库不随代码提供该权重，需要从学长/作者处取得，或者用官方 `trainer_flowpure.py` 训练。

## Colab 依赖

不要执行官方 `requirements.txt` 中固定版本的 `torch==2.3.1`，避免覆盖 Colab 当前 CUDA 版 PyTorch。只补充缺失包：

```python
!pip -q install fvcore ninja gdown
```

这里不安装 `torchattacks`：它会把 Colab 的 `requests` 和 `urllib3` 降级，而本评测脚本并不调用它。`foolbox`、`kornia` 和 `torchcfm` 只在官方的其他攻击/训练入口中使用，本评测同样不需要。

评测入口内置了 FlowPure 的 Python 3.13 兼容修复：官方 `random.randint(0, 1e8)` 的浮点上界会在新版 Python 报错，脚本会将其替换为整数上界，不需要手改官方仓库。

下载 DiffPure 的 Score-SDE 权重：

```python
!mkdir -p /content/FlowPure/resources/checkpoints/score_sde
!gdown 16_-Ahc6ImZV5ClUc0vM5Iivf8OJ1VSif \
  -O /content/FlowPure/resources/checkpoints/score_sde/checkpoint_c10.pth
```

## 生成正式 AutoAttack Standard 缓存

沿用旧缓存中的同一组 clean 图像与标签，只重新生成标准 AutoAttack：

```python
%run /content/diffpure_flowpure_benchmark/prepare_autoattack_cache.py \
  --zeropur-root /content/zeropur_workspace/official \
  --source-cache /content/drive/MyDrive/zeropur_research/cache/aa_n500_seed20261112.pt \
  --output /content/drive/MyDrive/zeropur_research/cache/aa_standard_n500_seed20261112.pt
```

后续正式评测把 `--cache` 指向 `aa_standard_n500_seed20261112.pt`。若只是先检查程序，也可以继续使用旧缓存，但表格中应写成“APGD-CE+DLR”，不要写成完整 AutoAttack Standard。

## 先跑 100 张检查

```python
%run /content/diffpure_flowpure_benchmark/eval_purifier.py \
  --method diffpure \
  --flowpure-root /content/FlowPure \
  --zeropur-root /content/zeropur_workspace/official \
  --cache /content/drive/MyDrive/zeropur_research/cache/aa_standard_n500_seed20261112.pt \
  --output /content/drive/MyDrive/zeropur_research/results/diffpure_resnet18_n100.json \
  --limit 100
```

确认无误后删除 `--limit 100` 跑固定 500 张。FlowPure 只需把 `--method` 改为 `flowpure`，并更换输出文件名。

随机净化器的严格检查可加 `--bpda-eot 5 --eval-eot 5`；这比单次 BPDA 更强，但耗时约增加五倍。

## 使用 ZeroPur ResNet18 训练 CIFAR-10 FlowPure

训练脚本不依赖 `torchattacks` 或 `torchcfm`。它复用 FlowPure 官方 CIFAR NCSN++，使用 ZeroPur ResNet18 在线生成PGD训练输入，并以官方 `sigma=0` Conditional Flow Matching目标训练。

先在临时目录跑10步冒烟测试：

```python
%run /content/diffpure_flowpure_benchmark/train_flowpure_cifar10.py \
  --flowpure-root /content/FlowPure \
  --zeropur-root /content/zeropur_workspace/official \
  --data-root /content/cifar10 \
  --output /content/flowpure_smoke/checkpoint_c10.pth \
  --max-steps 10 \
  --physical-batch-size 2 \
  --gradient-accumulation 2 \
  --log-every 1 \
  --val-every 5 \
  --save-every 5
```

确认显存和loss正常后，先跑500步小阶段估算速度和趋势：

```python
%run /content/diffpure_flowpure_benchmark/train_flowpure_cifar10.py \
  --flowpure-root /content/FlowPure \
  --zeropur-root /content/zeropur_workspace/official \
  --data-root /content/cifar10 \
  --output /content/drive/MyDrive/zeropur_research/checkpoints/flowpure_pgd_resnet18/checkpoint_c10.pth \
  --max-steps 500 \
  --physical-batch-size 2 \
  --gradient-accumulation 16 \
  --val-every 100 \
  --save-every 500
```

结果正常后把累计目标提高到2000步并添加 `--resume`；`--max-steps` 表示累计optimizer step，不是额外步数：

```python
%run /content/diffpure_flowpure_benchmark/train_flowpure_cifar10.py \
  --flowpure-root /content/FlowPure \
  --zeropur-root /content/zeropur_workspace/official \
  --data-root /content/cifar10 \
  --output /content/drive/MyDrive/zeropur_research/checkpoints/flowpure_pgd_resnet18/checkpoint_c10.pth \
  --max-steps 2000 \
  --physical-batch-size 2 \
  --gradient-accumulation 16 \
  --val-every 200 \
  --save-every 500 \
  --resume
```

评测前把Drive权重链接到官方加载路径：

```python
!mkdir -p /content/FlowPure/resources/checkpoints/flowpure_pgd
!ln -sf /content/drive/MyDrive/zeropur_research/checkpoints/flowpure_pgd_resnet18/checkpoint_c10.pth \
  /content/FlowPure/resources/checkpoints/flowpure_pgd/checkpoint_c10.pth
```


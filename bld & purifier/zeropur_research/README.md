# ZeroPur research

该目录与 `experiments/` 完全独立。第一阶段保持官方 ZeroPur 实现不变，只复现
CIFAR-10 的固定子集基线；后续局部图引导和高频共识改进均在本目录继续开发。

官方方法与代码：

- Paper: https://arxiv.org/abs/2406.03143
- Code: https://github.com/erhul/ZeroPur

## Colab：准备隔离环境

将本目录上传到 `/content/zeropur_research`，然后运行：

```python
%run -i "/content/zeropur_research/setup_colab.py" \
  --root "/content/zeropur_workspace"
```

脚本会将官方仓库克隆到 `/content/zeropur_workspace/official`，并在
`environment.json` 中记录实际 commit 与运行环境。不会修改官方源码。

## 第一次试跑：100张、PGD-20

```python
%run -i "/content/zeropur_research/run_cifar10_pgd_baseline.py" \
  --official-root "/content/zeropur_workspace/official" \
  --data-root "/content/cifar10" \
  --classifier-type all \
  --subset-size 100 \
  --batch-size 8 \
  --attack-steps 20 \
  --output "/content/drive/MyDrive/zeropur_research/results/cifar10_pgd20_n100_seed20261112.json"
```

结果同时报告：干净、PGD、单次模糊、Guided Shift、完整 ZeroPur，以及干净图像
经过 ZeroPur 后的准确率。确认流程正常后，将 `--subset-size` 改成 `1000`。

这一步使用非自适应 PGD-20，是可控的最小复现，不等同于论文最终 AutoAttack
结果。只有基线数值与官方趋势一致后，才运行完整 AutoAttack、BPDA 和 EOT。

## BiGuidePur：固定子集消融

`run_biguided_cifar10.py` 分两阶段运行。攻击阶段只运行一次并缓存对抗样本：

```python
%run -i "/content/zeropur_research/run_biguided_cifar10.py" \
  --stage attack \
  --official-root "/content/zeropur_workspace/official" \
  --cache "/content/drive/MyDrive/zeropur_research/cache/aa_n500_seed20261112.pt" \
  --output "/content/unused.json" \
  --subset-size 500
```

消融评测阶段复用缓存，不重新攻击：

```python
%run -i "/content/zeropur_research/run_biguided_cifar10.py" \
  --stage evaluate \
  --official-root "/content/zeropur_workspace/official" \
  --cache "/content/drive/MyDrive/zeropur_research/cache/aa_n500_seed20261112.pt" \
  --output "/content/drive/MyDrive/zeropur_research/results/biguide_ablation_n500_seed20261112.json" \
  --subset-size 500 \
  --batch-size 8
```

输出对比单次高斯模糊、官方 coarse、局部图引导、高频三噪声共识，以及双分支
互相引导融合。当前攻击仍以官方分类器为目标，属于非自适应评测；随机局部引导的
最终鲁棒性必须再由 BPDA/EOT 端到端攻击验证。

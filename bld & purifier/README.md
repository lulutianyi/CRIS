# D-Fire 对抗净化实验文件导航

当前脚本已按用途分组。Colab 上请保留同样的目录结构（`/content/core`、`/content/methods`、`/content/comparisons`、`/content/baselines`），不能只上传某一个比较脚本。模型权重、D-Fire 数据和共享 PGD 图像仍在 Colab/Drive，本地没有搬动。

| 目录 | 内容 |
| --- | --- |
| 根目录 | `gcn.ipynb`（GCN-Skip 主 notebook）、`prepare.ipynb`（环境/数据准备）、本导航 |
| `core/` | `shared_pgd_gcn_flowpure_compare.py`：统一 PGD、GCN/FlowPure 模型定义和完整比较的共享实现 |
| `methods/` | `freqpure_dfire.py`、`flowpure_dfire.py`、`zeropur_dfire.py`、`gcn_robust_taskaware.py`：各方法评测或微调 |
| `comparisons/` | `compare_gcn_freqpure_subset.py`、`compare_gcn_joint_freqpure_subset.py`、`run_three_subset_gcn_freqpure.py`：同图比较与三组 400 图重复实验 |
| `baselines/` | `purify_bld_fixed.py`：旧 BLD/二进制扩散基线，仍由 notebook 导入 |
| `notebooks/` | 已归档的早期基线和分析 notebook |
| `archive/backups/` | 历史备份，不作为当前运行入口 |

## Colab 新路径

以下仅是**入口路径**示例；原有参数（如 `--stage`、`--shared-root`、`--output-root`）照旧添加。旧的 `/content/脚本名.py` 路径不再适用。

```python
%run -i /content/methods/gcn_robust_taskaware.py
%run -i /content/methods/flowpure_dfire.py
%run /content/methods/freqpure_dfire.py --stage all --max-images 2
%run /content/core/shared_pgd_gcn_flowpure_compare.py --stage map
%run /content/comparisons/compare_gcn_freqpure_subset.py --help
%run /content/comparisons/compare_gcn_joint_freqpure_subset.py --help
%run /content/comparisons/run_three_subset_gcn_freqpure.py --help
%run /content/methods/zeropur_dfire.py --help
```

`gcn.ipynb` 和 `prepare.ipynb` 已把 `/content/baselines` 加到导入路径；`gcn.ipynb` 最后一个单元也已改为 `/content/methods/gcn_robust_taskaware.py`。如果 Colab 中仍只放着旧的扁平脚本，请先按新目录上传，否则会报找不到模块。`flowpure_dfire.py` 依赖 notebook 中已建立的变量，只能在对应 notebook 环境用 `%run -i`，不能独立 `!python` 运行。

这次只移动当前脚本、更新导入和路径说明，没有拆分模型实现或修改训练/评测算法，也没有删除实验结果或权重。较长的 `core/shared_pgd_gcn_flowpure_compare.py` 仍是原先的完整实现；真正拆小需要单独做输出一致性验证。

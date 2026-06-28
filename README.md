【项目背景】
CUHK-Shenzhen本科生，研究D-Fire数据集图像压缩+检测联合优化，
目标是为UAV场景下低带宽通信设计压缩方案。

【已完成的技术路线】
VQ-VAE基线(PSNR 23) → CompressAI(FactorizedPrior) → Cheng2020Anchor 
→ 任务联合损失(+YOLOv8检测特征损失) → 低bpp微调 → 简化版残差增强网络

【关键权重，存于 Drive: /content/drive/MyDrive/dfire_checkpoints/】
- cheng2020_task_aware_best.pth     bpp=0.42, PSNR=30.21
- cheng2020_low_bpp_best.pth        bpp=0.18, PSNR=30.35
- yolov8n_dfire_detector.pt         D-Fire检测器, mAP50=0.656(原图)
- residual_unet_low_bpp_best.pth    单步残差增强网络
- 配套脚本: residual_unet_model.py / train_residual_enhance.py / 
            eval_map_comparison.py / generate_rddm_pairs.py

【核心结论：四方mAP对比】
原图 baseline:              mAP50=0.656  mAP50-95=0.353
task-aware(bpp=0.42):       mAP50=0.425  mAP50-95=0.207  (-35.2%)
low-bpp(bpp=0.18,未增强):    mAP50=0.403  mAP50-95=0.194  (-38.6%)
low-bpp+残差增强:            mAP50=0.425  mAP50-95=0.209  (-35.2%, 挽回3.4个百分点)

【关键发现】
1. PSNR和检测mAP脱钩——两个压缩模型PSNR都~30dB，但mAP下降35~39%，
   说明压缩破坏的是检测器依赖的高频边缘信息，而PSNR感知不到这种损失。
2. 可视化诊断显示：检测框大多没有完全消失，而是边界定位发生漂移、
   置信度波动，说明问题更接近"细节模糊"而非"语义结构丢失"。
3. 简化版残差增强（单步UNet，非完整RDDM扩散）证明思路有效但能力有限：
   用相同bpp(0.18)追平了bpp=0.42才能达到的检测精度。

【下一步可选方向】
A. 增大残差网络容量（更深/更宽的UNet）
B. 改进残差网络的损失函数（加入检测特征损失，而不只是L1像素损失）
C. 尝试多步迭代精化（更接近完整RDDM的扩散思想）
D. 回头优化任务联合损失本身（增大μ权重，或换检测头损失而非特征L2距离）

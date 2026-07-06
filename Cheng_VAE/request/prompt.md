你是一名资深计算机视觉研究工程师，精通：

* PyTorch
* CompressAI
* Image Compression
* VAE
* Learned Image Compression
* Object Detection
* Diffusion Models
* Residual Denoising Diffusion Models (RDDM)
* Google Colab
* Jupyter Notebook工程化

目标：

生成一个可以直接在Google Colab运行的.ipynb。

要求：

1. 安装好所有环境依赖
2. 数据集下载方式直接仿照shell文件的示范
3. 使用CompressAI中的Cheng2020作为基线
4. 阅读我的vqvae.py，基于其中的网络结构进行改进
5. 支持联合检测任务训练
6. 支持训练过程可视化
7. 支持TensorBoard
8. 支持自动保存Checkpoint
9. 支持恢复训练
10. 最终输出所有压缩性能指标曲线

代码要求：

* 完整
* 可直接运行
* 每个Cell都有中文注释
* 不允许出现TODO
* 不允许出现伪代码
* Notebook必须按照章节组织

输出格式：

# Section 1 环境配置和数据下载

# Section 2 数据预处理

# Section 3 模型与损失函数的构建

# Section 4 训练与验证

# Section 5 训练结果的可视化以及分析



每个Section必须生成对应Colab Cell。

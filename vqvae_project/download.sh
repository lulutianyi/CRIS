# 1. 安装 Kaggle CLI
!pip install kaggle -q

# 2. 上传你的 kaggle.json（从 Kaggle → Account → Create API Token 下载）
from google.colab import files
files.upload()  # 选择你的 kaggle.json

# 3. 配置权限
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

# 4. 下载数据集
!kaggle datasets download -d shubhamkarande13/d-fire

# 5. 解压到指定文件夹
!unzip d-fire.zip -d /content/D-Fire
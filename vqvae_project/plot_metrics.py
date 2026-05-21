import numpy as np
import matplotlib.pyplot as plt

# ===== 加载数据 =====
loss = np.load("outputs/loss.npy")
psnr = np.load("outputs/psnr.npy")
mse = np.load("outputs/mse.npy")

# ===== Loss 曲线 =====
plt.figure()
plt.plot(loss)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.show()

# ===== PSNR 曲线 =====
plt.figure()
plt.plot(psnr)
plt.xlabel("Iteration")
plt.ylabel("PSNR")
plt.title("PSNR Curve")
plt.show()

# ===== MSE 曲线 =====
plt.figure()
plt.plot(mse)
plt.xlabel("Iteration")
plt.ylabel("MSE")
plt.title("MSE Curve")
plt.show()
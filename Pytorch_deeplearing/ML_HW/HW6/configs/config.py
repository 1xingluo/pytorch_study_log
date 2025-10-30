import torch
from torch.autograd import Variable
import os

class Config:
    def __init__(self):
        # ---- 基本设置 ----
        self.workspace_dir = r"D:\zl\ml2021-spring-hw6\crypko_data"
        self.batch_size = 64
        self.z_dim = 128
        self.lr = 5e-5
        self.n_epoch = 50
        self.n_critic = 5
        # self.clip_value = 0.01  # 可选参数（用于WGAN）

        # ---- 目录设置 ----
        self.log_dir = os.path.join(self.workspace_dir, 'logs')
        self.ckpt_dir = os.path.join(self.workspace_dir, 'checkpoints')

        # ---- 样本噪声 ----
        # 注意：这里创建固定采样噪声 z_sample，用于生成固定可视化样本
        self.z_sample = Variable(torch.randn(100, self.z_dim)).cuda()
                # ---- 设备设置 ----
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ---- 自动创建文件夹 ----
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.ckpt_dir, exist_ok=True)
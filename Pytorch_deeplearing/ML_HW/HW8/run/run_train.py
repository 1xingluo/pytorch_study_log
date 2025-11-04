import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from Config.config import Config,same_seeds
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from dataset.dataset import CustomTensorDataset
import torch
from model.model import fcn_autoencoder,conv_autoencoder,VAE,Resnet,loss_vae
import torch.nn as nn
from qqdm import qqdm, format_str
cfg=Config()
seed=cfg.seed
same_seeds(seed)
train = np.load('D:/zl/ml2021spring-hw8/ml2022spring-hw8/data/trainingset.npy', allow_pickle=True)
test = np.load('D:/zl/ml2021spring-hw8/ml2022spring-hw8/data/testingset.npy', allow_pickle=True)

print(train.shape)
print(test.shape)

# Training hyperparameters
num_epochs = cfg.num_epochs
batch_size = cfg.batch_size # medium: smaller batchsize
learning_rate = cfg.learning_rate
# Build training dataloader

x = torch.from_numpy(train)
train_dataset = CustomTensorDataset(x)

train_sampler = RandomSampler(train_dataset)
train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size)
print(len(train_dataloader))
# Model
model_type = cfg.model_type  # selecting a model type from {'cnn', 'fcn', 'vae', 'resnet'}
model_classes = {'resnet': Resnet(), 'fcn':fcn_autoencoder(), 'cnn':conv_autoencoder(), 'vae':VAE(), }
model = model_classes[model_type].cuda()

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(
    model.parameters(), lr=learning_rate)



# best_loss = float('inf')

# # 外层进度条：Epoch
# qqdm_epochs = qqdm(range(num_epochs), desc=format_str('bold', 'Training'))

# for epoch in qqdm_epochs:
#     model.train()
#     tot_loss = []

#     # 内层进度条：Iteration（每个 epoch 内的 batch）
#     qqdm_iters = qqdm(enumerate(train_dataloader), total=len(train_dataloader),
#                       desc=f'Epoch {epoch + 1}/{num_epochs}')

#     for step, data in qqdm_iters:
#         # =================== 数据加载 ======================
#         if model_type in ['cnn', 'vae', 'resnet']:
#             img = data.float().cuda(non_blocking=True)
#         elif model_type in ['fcn']:
#             img = data.float().cuda(non_blocking=True)
#             img = img.view(img.shape[0], -1)

#         # =================== 前向传播 ======================
#         output = model(img)
#         if model_type == 'vae':
#             recon, mu, logvar = output
#             loss = loss_vae(recon, img, mu, logvar, criterion)
#         else:
#             loss = criterion(output, img)

#         # =================== 反向传播 ======================
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         # =================== 统计与日志 ======================
#         tot_loss.append(loss.item())

#         qqdm_iters.set_infos({
#             'Iter': f'{step + 1}/{len(train_dataloader)}',
#             'Loss': f'{loss.item():.4f}',
#         })

#     # =================== 每个 epoch 结束 ======================
#     mean_loss = np.mean(tot_loss)

#     # 保存最佳模型
#     if mean_loss < best_loss:
#         best_loss = mean_loss
#         torch.save(model.state_dict(), f'best_model_{model_type}.pt')

#     # 更新外层进度条信息
#     qqdm_epochs.set_infos({
#         'Epoch': f'{epoch + 1}/{num_epochs}',
#         'MeanLoss': f'{mean_loss:.4f}',
#         'Best': f'{best_loss:.4f}',
#     })

# # =================== 训练结束后保存最终模型 ======================
# torch.save(model.state_dict(), f'last_model_{model_type}.pt')
# print("✅ Training complete. Best loss:", best_loss)



eval_batch_size = 200

# build testing dataloader
data = torch.tensor(test, dtype=torch.float32)
test_dataset = CustomTensorDataset(data)
test_sampler = SequentialSampler(test_dataset)
test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=eval_batch_size, num_workers=0)
eval_loss = nn.MSELoss(reduction='none')

# load trained model
checkpoint_path = 'last_model_cnn.pt'
from model.model import conv_autoencoder  # 确保导入定义

model = conv_autoencoder()
model.load_state_dict(torch.load(
    r'D:/python_code/Pytorch_deeplearing/best_model_cnn.pt',
    map_location='cpu'  # 先加载到CPU，避免跨设备错误
))
model = model.cuda()  # ✅ 移动到GPU
model.eval()

# prediction file 
out_file = 'PREDICTION_FILE.csv'



import pandas as pd  
anomality = list()
with torch.no_grad():
  for i, data in enumerate(test_dataloader): 
        if model_type in ['cnn', 'vae', 'resnet']:
            img = data.float().cuda()
        elif model_type in ['fcn']:
            img = data.float().cuda()
            img = img.view(img.shape[0], -1)
        else:
            img = data[0].cuda()
        output = model(img)
        if model_type in ['cnn', 'resnet', 'fcn']:
            output = output
        elif model_type in ['res_vae']:
            output = output[0]
        elif model_type in ['vae']: # , 'vqvae'
            output = output[0]
        if model_type in ['fcn']:
            loss = eval_loss(output, img).sum(-1)
        else:
            loss = eval_loss(output, img).sum([1, 2, 3])
        anomality.append(loss)
anomality = torch.cat(anomality, axis=0)
anomality = torch.sqrt(anomality).reshape(len(test), 1).cpu().numpy()

df = pd.DataFrame(anomality, columns=['Predicted'])
df.to_csv(out_file, index_label = 'Id')


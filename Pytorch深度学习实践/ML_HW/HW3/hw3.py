# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision.datasets import DatasetFolder
from tqdm.auto import tqdm
import math

# --------------------- 训练集增强 ---------------------
train_tfm = transforms.Compose([
    transforms.RandomResizedCrop(128, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.2),  # 加入垂直翻转
    transforms.RandomRotation(degrees=20),
    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.05),
    transforms.RandomGrayscale(p=0.1),      # 随机灰度
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.1, scale=(0.02, 0.1), ratio=(0.3, 3.3))  # 随机擦除
])

# --------------------- 验证/测试集 ---------------------
test_tfm = transforms.Compose([
    transforms.Resize(140),
    transforms.CenterCrop(128),  # 保留中心区域
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ============================================================
# 数据集加载
# ============================================================
train_dir = "D:/资料/ml2021spring-hw3/food-11/training/labeled"
valid_dir = "D:/资料/ml2021spring-hw3/food-11/validation"
unlabeled_dir = "D:/资料/ml2021spring-hw3/food-11/training/unlabeled"
test_dir = "D:/资料/ml2021spring-hw3/food-11/testing"

batch_size = 64

train_set = DatasetFolder(train_dir, loader=lambda x: Image.open(x).convert("RGB"), extensions="jpg", transform=train_tfm)
valid_set = DatasetFolder(valid_dir, loader=lambda x: Image.open(x).convert("RGB"), extensions="jpg", transform=test_tfm)
unlabeled_set = DatasetFolder(unlabeled_dir, loader=lambda x: Image.open(x).convert("RGB"), extensions="jpg", transform=train_tfm)
test_set = DatasetFolder(test_dir, loader=lambda x: Image.open(x).convert("RGB"), extensions="jpg", transform=test_tfm)

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True)
valid_loader = DataLoader(valid_set, batch_size=512, shuffle=True, pin_memory=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

# ============================================================
# 模型定义
# ============================================================
class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2),

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.15),

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.1),

            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2),

            nn.Conv2d(512, 1024, 3, 1, 1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.3),
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(1024 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 11)
        )

    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.flatten(1)
        x = self.fc_layers(x)
        return x

# ============================================================
# 半监督伪标签
# ============================================================
class PseudoDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        img = Image.open(img_path).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

def get_pseudo_labels(dataset, model, batch_size=64, threshold=0.8):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    softmax = nn.Softmax(dim=-1)

    pseudo_images = []
    pseudo_labels = []

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    for batch_idx, (imgs, _) in enumerate(tqdm(dataloader, desc="Pseudo-labeling")):
        imgs = imgs.to(device)
        with torch.no_grad():
            logits = model(imgs)
            probs = softmax(logits)
        confs, preds = probs.max(dim=-1)
        for i in range(len(imgs)):
            if confs[i] >= threshold:
                pseudo_images.append(dataset.samples[batch_idx * batch_size + i][0])
                pseudo_labels.append(preds[i].item())
    model.train()
    return PseudoDataset(pseudo_images, pseudo_labels, transform=dataset.transform)

# ============================================================
# 训练配置
# ============================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
model = Classifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0002, weight_decay=1e-5)

n_epochs = 150
warmup_epochs = 5
do_semi = True
best_valid_acc = 0.0
save_path = "best_model.pth"
last_pseudo_acc = 0.0  # 上一次生成伪标签时验证集准确率

# Cosine Annealing + Warmup 调度器
def lr_lambda(current_epoch):
    if current_epoch < warmup_epochs:
        return float(current_epoch + 1) / warmup_epochs
    else:
        progress = (current_epoch - warmup_epochs) / (n_epochs - warmup_epochs)
        return 0.5 * (1 + math.cos(math.pi * progress))
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

# ============================================================
# 主训练循环
# ============================================================
for epoch in range(n_epochs):
    # -------------------- 验证阶段 --------------------
    model.eval()
    valid_loss, valid_accs = [], []
    for imgs, labels in tqdm(valid_loader, desc=f"Epoch {epoch+1} [Valid]"):
        imgs, labels = imgs.to(device), labels.to(device)
        with torch.no_grad():
            logits = model(imgs)
            loss = criterion(logits, labels)
            acc = (logits.argmax(dim=-1) == labels).float().mean()
        valid_loss.append(loss.item())
        valid_accs.append(acc.item())

    avg_valid_loss = np.mean(valid_loss)
    avg_valid_acc = np.mean(valid_accs)
    print(f"[Valid] Epoch {epoch+1}/{n_epochs} | Loss: {avg_valid_loss:.5f} | Acc: {avg_valid_acc:.5f}")

    # -------------------- 半监督伪标签 --------------------
    if do_semi and avg_valid_acc > 0.5 and avg_valid_acc > last_pseudo_acc :
        print(f"\n[Epoch {epoch+1}] Generating pseudo labels...")
        pseudo_set = get_pseudo_labels(unlabeled_set, model)
        concat_dataset = ConcatDataset([train_set, pseudo_set])
        train_loader = DataLoader(concat_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
        print(f"Semi-supervised dataset size: {len(concat_dataset)}")
        last_pseudo_acc = avg_valid_acc  # 更新上一次生成伪标签的准确率

    # -------------------- 训练阶段 --------------------
    model.train()
    train_loss, train_accs = [], []
    for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]"):
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
        optimizer.step()
        acc = (logits.argmax(dim=-1) == labels).float().mean()
        train_loss.append(loss.item())
        train_accs.append(acc.item())

    avg_train_loss = np.mean(train_loss)
    avg_train_acc = np.mean(train_accs)
    print(f"[Train] Epoch {epoch+1}/{n_epochs} | Loss: {avg_train_loss:.5f} | Acc: {avg_train_acc:.5f}")

    # -------------------- 更新学习率 --------------------
    scheduler.step()
    current_lr = optimizer.param_groups[0]['lr']
    print(f"→ Learning rate after scheduler: {current_lr:.6f}")

    # -------------------- 保存最优模型 --------------------
    if avg_valid_acc > best_valid_acc:
        best_valid_acc = avg_valid_acc
        torch.save(model.state_dict(), save_path)
        print(f"✅ Best model saved at epoch {epoch+1} with acc = {best_valid_acc:.5f}")

# ============================================================
# 加载最优模型 & 测试集预测
# ============================================================
model.load_state_dict(torch.load(save_path))
model.eval()

predictions = []
for imgs, _ in tqdm(test_loader, desc="Testing"):
    imgs = imgs.to(device)
    with torch.no_grad():
        logits = model(imgs)
        preds = logits.argmax(dim=-1)
        predictions.extend(preds.cpu().numpy().tolist())

with open("predict.csv", "w") as f:
    f.write("Id,Category\n")
    for i, pred in enumerate(predictions):
        f.write(f"{i},{pred}\n")

print(f"Predictions saved to predict.csv ({len(predictions)} samples)")

# -*- coding: utf-8 -*-
import os
import math
import numpy as np
from PIL import Image
from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torchvision.transforms as transforms
from torchvision.datasets import DatasetFolder
from torchvision import models

# ============================================================
# 数据增强
# ============================================================
train_tfm = transforms.Compose([
    transforms.RandomResizedCrop(128, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.RandomRotation(degrees=20),
    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.05),
    transforms.RandomGrayscale(p=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.1, scale=(0.02, 0.1), ratio=(0.3, 3.3))
])

test_tfm = transforms.Compose([
    transforms.Resize(140),
    transforms.CenterCrop(128),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ============================================================
# 半监督伪标签 Dataset
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

def get_pseudo_labels(dataset, model, batch_size=64, threshold=0.8, device="cpu"):
    """
    关键改动：
    - 推理阶段临时把 dataset.transform 切到 test_tfm（弱增广），筛出高置信样本；
    - 返回的 PseudoDataset 用 train_tfm（强增广）参与训练。
    """
    model.eval()
    softmax = nn.Softmax(dim=-1)
    pseudo_images, pseudo_labels = [], []

    # ---- 临时把无标签数据集切到“弱增广视图”做推理 ----
    old_transform = dataset.transform
    dataset.transform = test_tfm
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    processed = 0
    with torch.no_grad():
        for (imgs, _) in tqdm(dataloader, desc="Pseudo-labeling (weak aug)"):
            imgs = imgs.to(device)
            logits = model(imgs)
            probs = softmax(logits)
            confs, preds = probs.max(dim=-1)

            bs = imgs.size(0)
            for i in range(bs):
                if confs[i].item() >= threshold:
                    # 用原始 dataset 的文件路径（DatasetFolder: dataset.samples[idx][0]）
                    idx = processed + i
                    pseudo_images.append(dataset.samples[idx][0])
                    pseudo_labels.append(preds[i].item())
            processed += bs

    # 还原 transform，避免影响后续使用
    dataset.transform = old_transform
    model.train()

    # ---- 训练阶段对伪标签样本使用强增广 ----
    return PseudoDataset(pseudo_images, pseudo_labels, transform=train_tfm)

# ============================================================
# ResNet18 分类模型
# ============================================================
class Classifier(nn.Module):
    def __init__(self, num_classes=11, pretrained=False):
        super().__init__()
        self.backbone = models.resnet18(weights=None if not pretrained else models.ResNet18_Weights.IMAGENET1K_V1)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)

# ============================================================
# Windows 兼容主程序
# ============================================================
if __name__ == "__main__":
    # -------------------- 数据集 --------------------
    train_dir = "D:/资料/ml2021spring-hw3/food-11/training/labeled"
    valid_dir = "D:/资料/ml2021spring-hw3/food-11/validation"
    unlabeled_dir = "D:/资料/ml2021spring-hw3/food-11/training/unlabeled"
    test_dir = "D:/资料/ml2021spring-hw3/food-11/testing"

    batch_size = 64

    # 仅此处小修：extensions 必须是元组，而不是字符串
    exts = ('.jpg', '.jpeg', '.png')

    train_set = DatasetFolder(train_dir, loader=lambda x: Image.open(x).convert("RGB"),
                              extensions=exts, transform=train_tfm)
    valid_set = DatasetFolder(valid_dir, loader=lambda x: Image.open(x).convert("RGB"),
                              extensions=exts, transform=test_tfm)
    unlabeled_set = DatasetFolder(unlabeled_dir, loader=lambda x: Image.open(x).convert("RGB"),
                                  extensions=exts, transform=train_tfm)  # 训练时仍用强增广
    test_set = DatasetFolder(test_dir, loader=lambda x: Image.open(x).convert("RGB"),
                             extensions=exts, transform=test_tfm)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=0)
    valid_loader = DataLoader(valid_set, batch_size=512, shuffle=False, pin_memory=True, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

    # -------------------- 模型、损失、优化器 --------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Classifier(num_classes=11, pretrained=False).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4, weight_decay=1e-5)

    # -------------------- 训练参数 --------------------
    n_epochs = 300
    warmup_epochs = 5
    pseudo_label_interval = 5
    do_semi = True
    best_valid_acc = 0.0
    save_path = "best_model.pth"

    # -------------------- 早停参数 --------------------
    early_stop_patience = 30  # 验证集连续多少轮没有提升就停止
    epochs_no_improve = 0

    # Cosine Annealing + Warmup 调度器
    def lr_lambda(current_epoch):
        if current_epoch < warmup_epochs:
            return float(current_epoch + 1) / warmup_epochs
        else:
            progress = (current_epoch - warmup_epochs) / (n_epochs - warmup_epochs)
            return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    # -------------------- 主训练循环 --------------------
    for epoch in range(n_epochs):
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

        # -------------------- 检查早停 --------------------
        if avg_valid_acc > best_valid_acc:
            best_valid_acc = avg_valid_acc
            torch.save(model.state_dict(), save_path)
            epochs_no_improve = 0
            print(f"✅ Best model saved at epoch {epoch+1} with acc = {best_valid_acc:.5f}")
        else:
            epochs_no_improve += 1
            print(f"⚠️ No improvement for {epochs_no_improve} epochs.")

        if epochs_no_improve >= early_stop_patience:
            print(f"⏹ Early stopping triggered at epoch {epoch+1}")
            break

        # -------------------- 半监督伪标签（每 N 轮） --------------------
        if do_semi and ((epoch + 1) % pseudo_label_interval == 0):
            print(f"\n[Epoch {epoch+1}] Generating pseudo labels (weak aug infer, strong aug train)...")
            pseudo_set = get_pseudo_labels(unlabeled_set, model, batch_size=batch_size, threshold=0.8, device=device)
            concat_dataset = ConcatDataset([train_set, pseudo_set])
            train_loader = DataLoader(concat_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=0)
            print(f"Semi-supervised dataset size: {len(concat_dataset)}")

        # -------------------- 学习率更新 --------------------
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        print(f"→ Learning rate after scheduler: {current_lr:.6f}")

    # -------------------- 测试集预测 --------------------
    model.load_state_dict(torch.load(save_path, map_location=device))
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

    print(f"✅ Predictions saved to predict.csv ({len(predictions)} samples)")

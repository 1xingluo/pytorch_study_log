import os
import json
import torch
import random
from pathlib import Path
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import math
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.optim import AdamW
 #mapping是600个人，以及对应的id
#建立数据集：
class myDataset(Dataset):
  def __init__(self, data_dir, segment_len=128):
    self.data_dir = data_dir
    self.segment_len = segment_len
 
    # Load the mapping from speaker neme to their corresponding id. 
    mapping_path = Path(data_dir) / "mapping.json"
    mapping = json.load(mapping_path.open())
    self.speaker2id = mapping["speaker2id"]
 
    #self.speaker2id = mapping["speaker2id"]
    # 字典："speaker2id": {
    #     "id10001": 0,
    #     "id10002": 1,
    # }
    # Load metadata of training data.
    metadata_path = Path(data_dir) / "metadata.json"
    metadata = json.load(open(metadata_path))["speakers"]
 #metadata也是字典：
#  {
#     "speakers": {
#         "id10001": [
#             {"feature_path": "uttr-5c88b2f1803449789c36f14fb4d3c1eb.pt", "mel_len": 652},
#             {"feature_path": "uttr-xxxxx.pt", "mel_len": 743},
#             ...
#         ],
#         "id10002": [
#             {"feature_path": "...", "mel_len": ...},
#             ...
#         ]
#     }
# }

    # Get the total number of speaker.
    self.speaker_num = len(metadata.keys())
    self.data = []
    for speaker in metadata.keys():
      for utterances in metadata[speaker]:
        self.data.append([utterances["feature_path"], self.speaker2id[speaker]])
 
  def __len__(self):
    return len(self.data)
 
  def __getitem__(self, index):
    feat_path, speaker = self.data[index]
    # Load preprocessed mel-spectrogram.
    mel = torch.load(os.path.join(self.data_dir, feat_path))
 
    # Segmemt mel-spectrogram into "segment_len" frames.
    if len(mel) > self.segment_len:
      # Randomly get the starting point of the segment.
      start = random.randint(0, len(mel) - self.segment_len)
      # Get a segment with "segment_len" frames.
      mel = torch.FloatTensor(mel[start:start+self.segment_len])
    else:
      mel = torch.FloatTensor(mel)
    # Turn the speaker id into long for computing loss later.
    speaker = torch.FloatTensor([speaker]).long()
    return mel, speaker
 
  def get_speaker_number(self):
    return self.speaker_num


#datalolader


def collate_batch(batch):
  # Process features within a batch.
  """Collate a batch of data."""
  mel, speaker = zip(*batch)
  # Because we train the model batch by batch, we need to pad the features in the same batch to make their lengths the same.
  mel = pad_sequence(mel, batch_first=True, padding_value=-20)    # pad log 10^(-20) which is very small value.
  # mel: (batch size, length, 40)
  return mel, torch.FloatTensor(speaker).long()


def get_dataloader(data_dir, batch_size, n_workers):
  """Generate dataloader"""
  dataset = myDataset(data_dir)
  speaker_num = dataset.get_speaker_number()
  # Split dataset into training dataset and validation dataset
  trainlen = int(0.9 * len(dataset))
  lengths = [trainlen, len(dataset) - trainlen]
  trainset, validset = random_split(dataset, lengths)

  train_loader = DataLoader(
    trainset,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
    num_workers=n_workers,
    pin_memory=False,
    collate_fn=collate_batch,
  )
  valid_loader = DataLoader(
    validset,
    batch_size=batch_size,
    num_workers=n_workers,
    drop_last=True,
    pin_memory=False,
    collate_fn=collate_batch,
  )

  return train_loader, valid_loader, speaker_num

import torch
import torch.nn as nn
import torch.nn.functional as F

# --------- 前馈层 ---------
class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

# --------- 卷积模块 ---------
class ConformerConvModule(nn.Module):
    def __init__(self, dim, expansion_factor=2, kernel_size=31, dropout=0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(dim)
        self.pointwise1 = nn.Linear(dim, dim * expansion_factor)
        self.glu = nn.GLU(dim=-1)  # 门控线性单元
        self.depthwise_conv = nn.Conv1d(dim, dim, kernel_size, padding=kernel_size//2, groups=dim)
        self.batch_norm = nn.BatchNorm1d(dim)
        self.swish = nn.SiLU()  # 激活函数
        self.pointwise2 = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [seq_len, batch, dim]
        x = self.layer_norm(x)
        x = self.pointwise1(x)
        x = self.glu(x)
        # 转为 (batch, dim, seq_len) 方便 Conv1d
        x = x.transpose(0, 1).transpose(1, 2)
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = self.swish(x)
        x = x.transpose(1, 2).transpose(0, 1)
        x = self.pointwise2(x)
        x = self.dropout(x)
        return x
# --------- ConformerBlock ---------
class ConformerBlock(nn.Module):
    def __init__(self, dim, ff_mult=4, heads=2, conv_expansion=2,
                 conv_kernel=31, attn_dropout=0.1, ff_dropout=0.1, conv_dropout=0.1):
        super().__init__()
        self.ff1 = FeedForward(dim, mult=ff_mult, dropout=ff_dropout)
        self.attn = nn.MultiheadAttention(dim, num_heads=heads, dropout=attn_dropout)
        self.conv = ConformerConvModule(dim, expansion_factor=conv_expansion,
                                        kernel_size=conv_kernel, dropout=conv_dropout)
        self.ff2 = FeedForward(dim, mult=ff_mult, dropout=ff_dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        # x: [seq_len, batch, dim]
        x = x + 0.5 * self.ff1(x)      # 前馈层1
        x = x + self.attn(x, x, x)[0]  # 自注意力
        x = x + self.conv(x)           # 卷积模块
        x = x + 0.5 * self.ff2(x)      # 前馈层2
        x = self.norm(x)
        return x


# -----------------------
# Classifier
# -----------------------
class Classifier(nn.Module):
    def __init__(self, d_model=128, n_spks=600, dropout=0.1):
        super().__init__()
        self.prenet = nn.Linear(40, d_model)

        # 使用 ConformerBlock 替换 Transformer
        self.encoder_layer = ConformerBlock(
            dim=d_model,
            ff_mult=4,        # 前馈层扩展倍数
            heads=4,          # 注意力头数
            conv_expansion=2,
            conv_kernel=31,
            attn_dropout=dropout,
            ff_dropout=dropout,
            conv_dropout=dropout
        )

        # 如果要堆叠多层 Conformer
        # self.encoder = nn.ModuleList([ConformerBlock(dim=d_model, ff_mult=4, heads=1) for _ in range(2)])

        self.pred_layer = nn.Linear(d_model, n_spks)

    def forward(self, mels):
        # mels: (batch, len, 40)
        out = self.prenet(mels)        # (B, L, D)
        out = out.permute(1, 0, 2)     # (L, B, D)
        out = self.encoder_layer(out)  # (L, B, D)

        # 如果堆叠多层
        # for layer in self.encoder:
        #     out = layer(out)

        out = out.transpose(0, 1)      # (B, L, D)
        stats = out.mean(dim=1)        # mean pooling
        out = self.pred_layer(stats)   # (B, n_spks)
        return out




def get_cosine_schedule_with_warmup(
  optimizer: Optimizer,
  num_warmup_steps: int,
  num_training_steps: int,
  num_cycles: float = 0.5,
  last_epoch: int = -1,
):
  """
  Create a schedule with a learning rate that decreases following the values of the cosine function between the
  initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
  initial lr set in the optimizer.

  Args:
    optimizer (:class:`~torch.optim.Optimizer`):
      The optimizer for which to schedule the learning rate.
    num_warmup_steps (:obj:`int`):
      The number of steps for the warmup phase.
    num_training_steps (:obj:`int`):
      The total number of training steps.
    num_cycles (:obj:`float`, `optional`, defaults to 0.5):
      The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
      following a half-cosine).
    last_epoch (:obj:`int`, `optional`, defaults to -1):
      The index of the last epoch when resuming training.

  Return:
    :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
  """

  def lr_lambda(current_step):
    # Warmup
    if current_step < num_warmup_steps:
      return float(current_step) / float(max(1, num_warmup_steps))
    # decadence
    progress = float(current_step - num_warmup_steps) / float(
      max(1, num_training_steps - num_warmup_steps)
    )
    return max(
      0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
    )

  return LambdaLR(optimizer, lr_lambda, last_epoch)



def model_fn(batch, model, criterion, device):
  """Forward a batch through the model."""

  mels, labels = batch
  mels = mels.to(device)
  labels = labels.to(device)

  outs = model(mels)

  loss = criterion(outs, labels)

  # Get the speaker id with highest probability.
  preds = outs.argmax(1)
  # Compute accuracy.
  accuracy = torch.mean((preds == labels).float())

  return loss, accuracy



def valid(dataloader, model, criterion, device): 
  """Validate on validation set."""

  model.eval()
  running_loss = 0.0
  running_accuracy = 0.0
  pbar = tqdm(total=len(dataloader.dataset), ncols=0, desc="Valid", unit=" uttr")

  for i, batch in enumerate(dataloader):
    with torch.no_grad():
      loss, accuracy = model_fn(batch, model, criterion, device)
      running_loss += loss.item()
      running_accuracy += accuracy.item()

    pbar.update(dataloader.batch_size)
    pbar.set_postfix(
      loss=f"{running_loss / (i+1):.2f}",
      accuracy=f"{running_accuracy / (i+1):.2f}",
    )

  pbar.close()
  model.train()

  return running_accuracy / len(dataloader)


def parse_args():
  """arguments"""
  config = {
    "data_dir": "C:/Users/xingluo/Downloads/Dataset/Dataset",
    "save_path": "model.ckpt",
    "batch_size": 32,
    "n_workers": 0,
    "valid_steps": 2000,
    "warmup_steps": 1000,
    "save_steps": 10000,
    "total_steps": 70000,
  }

  return config


def main(
  data_dir,
  save_path,
  batch_size,
  n_workers,
  valid_steps,
  warmup_steps,
  total_steps,
  save_steps,
):
  """Main function."""
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"[Info]: Use {device} now!")

  train_loader, valid_loader, speaker_num = get_dataloader(data_dir, batch_size, n_workers)
  train_iterator = iter(train_loader)
  print(f"[Info]: Finish loading data!",flush = True)

  model = Classifier(n_spks=speaker_num).to(device)
  criterion = nn.CrossEntropyLoss()
  optimizer = AdamW(model.parameters(), lr=1e-3)
  scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
  print(f"[Info]: Finish creating model!",flush = True)

  best_accuracy = -1.0
  best_state_dict = None

  pbar = tqdm(total=valid_steps, ncols=0, desc="Train", unit=" step")

  for step in range(total_steps):
    # Get data
    try:
      batch = next(train_iterator)
    except StopIteration:
      train_iterator = iter(train_loader)
      batch = next(train_iterator)

    loss, accuracy = model_fn(batch, model, criterion, device)
    batch_loss = loss.item()
    batch_accuracy = accuracy.item()

    # Updata model
    loss.backward()
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
    
    # Log
    pbar.update()
    pbar.set_postfix(
      loss=f"{batch_loss:.2f}",
      accuracy=f"{batch_accuracy:.2f}",
      step=step + 1,
    )

    # Do validation
    if (step + 1) % valid_steps == 0:
      pbar.close()

      valid_accuracy = valid(valid_loader, model, criterion, device)

      # keep the best model
      if valid_accuracy > best_accuracy:
        best_accuracy = valid_accuracy
        best_state_dict = model.state_dict()

      pbar = tqdm(total=valid_steps, ncols=0, desc="Train", unit=" step")

    # Save the best model so far.
    if (step + 1) % save_steps == 0 and best_state_dict is not None:
      torch.save(best_state_dict, save_path)
      pbar.write(f"Step {step + 1}, best model saved. (accuracy={best_accuracy:.4f})")

  pbar.close()


if __name__ == "__main__":
  main(**parse_args())
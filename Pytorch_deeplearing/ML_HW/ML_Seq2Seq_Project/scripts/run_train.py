import sys
from pathlib import Path
import os
import logging
from argparse import Namespace
import torch
import torch.optim as optim
from fairseq.tasks.translation import TranslationTask
from fairseq import utils

# -----------------------------
# 将项目根目录加入 sys.path
# -----------------------------
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# -----------------------------
# 从项目导入模块
# -----------------------------
from train_epoch import train_one_epoch
from checkpoint_utils import validate_and_save, try_load_checkpoint
from configs.config import (
    SEED, DATA_BIN_DIR,
    max_tokens, accum_steps, num_workers,
    start_epoch, max_epoch, resume, savedir,
    keep_last_epochs, use_wandb,
    source_lang, target_lang
)
from model.transformer_model import build_transformer_model, default_model_args
from model.criterion import LabelSmoothedCrossEntropyCriterion
from model.optim import NoamOpt

# -----------------------------
# Logging
# -----------------------------
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger("seq2seq-demo")

# -----------------------------
# CUDA
# -----------------------------
cuda_env = utils.CudaEnvironment()
utils.CudaEnvironment.pretty_print_cuda_env_list([cuda_env])
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# -----------------------------
# 构建 Namespace，适配 fairseq
# -----------------------------
task_args = Namespace(
    data=str(DATA_BIN_DIR),
    source_lang=source_lang,
    target_lang=target_lang,
    train_subset='train',
    required_seq_len_multiple=8,
    dataset_impl='mmap',
    upsample_primary=1,
    left_pad_source=True,
    left_pad_target=False,
    max_source_positions=1024,
    max_target_positions=1024,
    load_alignments=False,
    truncate_source=False,
    num_batch_buckets=0,
)

# -----------------------------
# 初始化任务
# -----------------------------
task = TranslationTask.setup_task(task_args)

# -----------------------------
# 加载数据集
# -----------------------------
logger.info("Loading data for epoch 1...")
task.load_dataset(split="train", epoch=1)
task.load_dataset(split="valid", epoch=1)

# -----------------------------
# 构建数据迭代器函数
# -----------------------------
def load_data_iterator(task, split, epoch=1, max_tokens=max_tokens, num_workers=num_workers, cached=True):
    return task.get_batch_iterator(
        dataset=task.dataset(split),
        max_tokens=max_tokens,
        max_sentences=None,
        max_positions=utils.resolve_max_positions(task.max_positions(), max_tokens),
        ignore_invalid_inputs=True,
        seed=SEED,
        num_workers=num_workers,
        epoch=epoch,
        disable_iterator_cache=not cached,
    )

# -----------------------------
# 构建模型
# -----------------------------
model = build_transformer_model(default_model_args, task.source_dictionary, task.target_dictionary)
criterion = LabelSmoothedCrossEntropyCriterion(
    smoothing=0.1,
    ignore_index=task.target_dictionary.pad(),
)
base_optimizer = optim.AdamW(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9, weight_decay=0.0001)
optimizer = NoamOpt(
    model_size=default_model_args.encoder_embed_dim,
    factor=2,
    warmup=4000,
    optimizer=base_optimizer
)

model = model.to(device=device)
criterion = criterion.to(device=device)

# -----------------------------
# 打印模型信息
# -----------------------------
logger.info("task: {}".format(task.__class__.__name__))
logger.info("encoder: {}".format(model.encoder.__class__.__name__))
logger.info("decoder: {}".format(model.decoder.__class__.__name__))
logger.info("criterion: {}".format(criterion.__class__.__name__))
logger.info("optimizer: {}".format(optimizer.__class__.__name__))
logger.info(
    "num. model params: {:,} (num. trained: {:,})".format(
        sum(p.numel() for p in model.parameters()),
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    )
)
logger.info(f"max tokens per batch = {max_tokens}, accumulate steps = {accum_steps}")

# -----------------------------
# 构建训练迭代器
# -----------------------------
epoch_itr = load_data_iterator(task, "train", start_epoch)

# -----------------------------
# 从 checkpoint 恢复
# -----------------------------
try_load_checkpoint(model, optimizer, name=resume, logger=logger)

# -----------------------------
# 训练循环
# -----------------------------
while epoch_itr.next_epoch_idx <= max_epoch:
    train_one_epoch(
        epoch_itr, model, task, criterion, optimizer, accum_steps,
        device=device, logger=logger, config=None, use_wandb=use_wandb
    )
    
    validate_and_save(
        model, task, criterion, optimizer, epoch=epoch_itr.epoch,
        logger=logger, device=device, utils=utils,
        load_data_iterator_fn=load_data_iterator,
        max_tokens=max_tokens, num_workers=num_workers,
        target_lang=target_lang, use_wandb=use_wandb
    )
    
    logger.info("end of epoch {}".format(epoch_itr.epoch))
    
    # 构建下一轮 epoch 的迭代器
    epoch_itr = load_data_iterator(task, "train", epoch_itr.next_epoch_idx)

# -----------------------------
# 平均 checkpoint
# -----------------------------
checkdir = Path(savedir).absolute()
avg_checkpoint = checkdir / "avg_last_5_checkpoint.pt"
os.system(
    f'python "{Path("fairseq/scripts/average_checkpoints.py")}" '
    f'--inputs "{checkdir}" '
    f'--num-epoch-checkpoints 5 '
    f'--output "{avg_checkpoint}"'
)

import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import numpy as np
from fairseq import utils
from fairseq.data import iterators
from torch.cuda.amp import GradScaler, autocast

def train_one_epoch(epoch_itr, model, task, criterion, optimizer, accum_steps=1,
                    device='cuda', logger=None, config=None, use_wandb=False, wandb=None):
    itr = epoch_itr.next_epoch_itr(shuffle=True)
    itr = iterators.GroupedIterator(itr, accum_steps)

    stats = {"loss": []}
    scaler = GradScaler()
    model.train()
    progress = tqdm.tqdm(itr, desc=f"train epoch {epoch_itr.epoch}", leave=False)

    for samples in progress:
        model.zero_grad()
        accum_loss = 0
        sample_size = 0

        for i, sample in enumerate(samples):
            sample = utils.move_to_cuda(sample, device=device)
            target = sample["target"]
            sample_size_i = sample["ntokens"]
            sample_size += sample_size_i

            with autocast():
                net_output = model.forward(**sample["net_input"])
                lprobs = F.log_softmax(net_output[0], dim=-1)
                loss = criterion(lprobs.view(-1, lprobs.size(-1)), target.view(-1))
                accum_loss += loss.item()
                scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        optimizer.multiply_grads(1 / (sample_size or 1.0))
        gnorm = nn.utils.clip_grad_norm_(model.parameters(), config.clip_norm if config else 1.0)
        scaler.step(optimizer)
        scaler.update()

        loss_print = accum_loss / sample_size
        stats["loss"].append(loss_print)
        progress.set_postfix(loss=loss_print)

        if use_wandb:
            wandb.log({"train/loss": loss_print, "train/grad_norm": gnorm.item(),
                       "train/lr": optimizer.rate(), "train/sample_size": sample_size})

    loss_print = np.mean(stats["loss"])
    if logger: logger.info(f"training loss: {loss_print:.4f}")
    return stats

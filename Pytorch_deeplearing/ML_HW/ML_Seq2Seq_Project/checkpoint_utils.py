import torch
import shutil
from pathlib import Path
from configs.config import (
    max_tokens, accum_steps, num_workers,
    start_epoch, max_epoch, resume, savedir,
    keep_last_epochs, use_wandb,
    source_lang, target_lang
)
from validate import validate  # 注意：此处只引用 validate，不引用 run_train

class Config:
    max_tokens = max_tokens
    accum_steps = accum_steps
    num_workers = num_workers
    start_epoch = start_epoch
    max_epoch = max_epoch
    resume = resume
    savedir = savedir
    keep_last_epochs = keep_last_epochs
    use_wandb = use_wandb
    source_lang = source_lang
    target_lang = target_lang

config = Config()

def validate_and_save(model, task, criterion, optimizer, epoch, 
                      save=True, logger=None, device=None, utils=None,
                      load_data_iterator_fn=None, max_tokens=None, num_workers=None,
                      target_lang=None, use_wandb=False):
    """调用 validate 并保存 checkpoint"""
    # -----------------------------
    # 使用默认值，如果没有传入
    max_tokens = max_tokens or config.max_tokens
    num_workers = num_workers or config.num_workers
    target_lang = target_lang or config.target_lang
    # -----------------------------

    stats = validate(model, task, criterion, logger=logger, device=device, utils=utils,
                     load_data_iterator_fn=load_data_iterator_fn,
                     max_tokens=max_tokens, num_workers=num_workers,
                     target_lang=target_lang, use_wandb=use_wandb)
    bleu = stats['bleu']
    loss = stats['loss']

    if save:
        savedir_path = Path(config.savedir).absolute()
        savedir_path.mkdir(parents=True, exist_ok=True)

        check = {
            "model": model.state_dict(),
            "stats": {"bleu": bleu.score, "loss": loss},
            "optim": {"step": getattr(optimizer, "_step", 0)}
        }

        torch.save(check, savedir_path / f"checkpoint{epoch}.pt")
        shutil.copy(savedir_path / f"checkpoint{epoch}.pt", savedir_path / "checkpoint_last.pt")

        if logger:
            logger.info(f"saved epoch checkpoint: {savedir_path}/checkpoint{epoch}.pt")

        # 保存示例文本
        with open(savedir_path / f"samples{epoch}.{config.source_lang}-{config.target_lang}.txt", "w", encoding="utf-8") as f:
            for s, h in zip(stats["srcs"], stats["hyps"]):
                f.write(f"{s}\t{h}\n")

        # 保存最佳 BLEU
        if getattr(validate_and_save, "best_bleu", 0) < bleu.score:
            validate_and_save.best_bleu = bleu.score
            torch.save(check, savedir_path / "checkpoint_best.pt")

        # 删除过期 checkpoint
        del_file = savedir_path / f"checkpoint{epoch - config.keep_last_epochs}.pt"
        if del_file.exists():
            del_file.unlink()

    return stats


def try_load_checkpoint(model, optimizer=None, name=None, logger=None):
    """加载 checkpoint"""
    name = name if name else "checkpoint_last.pt"
    checkpath = Path(config.savedir) / name
    if checkpath.exists():
        check = torch.load(checkpath, map_location='cpu')
        model.load_state_dict(check["model"])
        stats = check["stats"]
        step = getattr(optimizer, "_step", "unknown")
        if optimizer is not None:
            optimizer._step = check["optim"].get("step", step)
        if logger:
            logger.info(f"loaded checkpoint {checkpath}: step={step} loss={stats['loss']} bleu={stats['bleu']}")
        return stats
    else:
        if logger:
            logger.info(f"no checkpoints found at {checkpath}!")
        return None

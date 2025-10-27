import torch
import torch.nn.functional as F
import tqdm
import numpy as np
import sacrebleu
from argparse import Namespace


def validate(model, task, criterion, logger=None, device='cuda', utils=None,
             load_data_iterator_fn=None, max_tokens=4000, num_workers=0,
             target_lang='zh', use_wandb=False, beam=5, max_len_a=1.2, max_len_b=10):
    """
    Validation 流程，支持 beam search。
    model: nn.Module
    task: fairseq translation task
    criterion: loss function
    load_data_iterator_fn: 返回 batch iterator 的函数
    """

    if logger:
        logger.info('Begin validation with beam search...')

    # ----------- 构建验证数据迭代器 -----------
    itr = load_data_iterator_fn(
        task, "valid", 1, max_tokens=max_tokens, num_workers=num_workers
    ).next_epoch_itr(shuffle=False)

    stats = {"loss": [], "bleu": 0, "srcs": [], "hyps": [], "refs": []}
    srcs, hyps, refs = [], [], []

    model.eval()

    # ----------- beam search generator 参数 -----------
    gen_args = Namespace(
        beam=beam,
        max_len_a=max_len_a,
        max_len_b=max_len_b,
        stop_early=True,
        len_penalty=1.0,
        unk_penalty=0.0,
        sampling=False,
        sampling_topk=-1,
        sampling_topp=-1,
        temperature=1.0,
        diverse_beam_groups=-1,
        diverse_beam_strength=0.5,
        match_source_len=False,
        no_repeat_ngram_size=0,
        retain_dropout=False,
    )
    sequence_generator = task.build_generator([model], gen_args)
    # -------------------------------------------

    progress = tqdm.tqdm(itr, desc="validation", leave=False)

    with torch.no_grad():
        for sample in progress:
            sample = utils.move_to_cuda(sample, device=device)

            # ---------------- loss 计算 ----------------
            net_output = model(**sample["net_input"])
            lprobs = F.log_softmax(net_output[0], -1)
            target = sample["target"]
            sample_size = sample["ntokens"]
            loss = criterion(lprobs.view(-1, lprobs.size(-1)), target.view(-1)) / sample_size
            stats["loss"].append(loss)
            # ------------------------------------------

            # ---------------- beam search 生成 ----------------
            translations = task.inference_step(sequence_generator, [model], sample)

            for i, t in enumerate(translations):
                # 取 beam=1 的结果（最优翻译）
                hyp_tokens = t[0]["tokens"].int().cpu()
                hyp_str = task.target_dictionary.string(hyp_tokens).replace("@@ ", "")

                # 获取源句与参考句
                src_tokens = sample["net_input"]["src_tokens"][i]
                tgt_tokens = sample["target"][i]

                src_str = task.source_dictionary.string(
                    utils.strip_pad(src_tokens, task.source_dictionary.pad())
                ).replace("@@ ", "")
                ref_str = task.target_dictionary.string(
                    utils.strip_pad(tgt_tokens, task.target_dictionary.pad())
                ).replace("@@ ", "")

                srcs.append(src_str)
                hyps.append(hyp_str)
                refs.append(ref_str)
            # -------------------------------------------------

    # 计算平均 loss
    stats["loss"] = torch.stack(stats["loss"]).mean().item()

    # 计算 BLEU
    stats["bleu"] = sacrebleu.corpus_bleu(
        hyps, [refs],
        tokenize='zh' if target_lang == 'zh' else '13a'
    )

    stats["srcs"], stats["hyps"], stats["refs"] = srcs, hyps, refs

    # ---------------- 日志与 WandB ----------------
    if use_wandb:
        import wandb
        wandb.log({"valid/loss": stats["loss"], "valid/bleu": stats["bleu"].score}, commit=False)

    showid = np.random.randint(len(hyps))
    if logger:
        logger.info("Example source: " + srcs[showid])
        logger.info("Example hypothesis: " + hyps[showid])
        logger.info("Example reference: " + refs[showid])
        logger.info(f"Validation loss: {stats['loss']:.4f}")
        logger.info(stats["bleu"].format())

    return stats

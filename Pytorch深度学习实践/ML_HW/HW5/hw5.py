import os
import random
import shutil
import re
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
import sentencepiece as spm
from types import SimpleNamespace as Namespace
import logging
import sys
import torch.nn.functional as F
# -----------------------------
# 设置随机种子
# -----------------------------
seed = 73
random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# -----------------------------
# 数据路径配置
# -----------------------------
data_root = Path(r'D:\zl\ml2021spring-hw5')
dataset_name = 'ted2020'
ted_dir = data_root / dataset_name
test_dir = data_root / 'test'

# 原始文件 & 目标文件
raw_files = {
    'train_en': ted_dir / 'raw.en',
    'train_zh': ted_dir / 'raw.zh',
    'test_en': test_dir / 'test.en',
    'test_zh': test_dir / 'test.zh'
}

target_files = {
    'train_en': ted_dir / 'train_dev.raw.en',
    'train_zh': ted_dir / 'train_dev.raw.zh',
    'test_en': test_dir / 'test.raw.en',
    'test_zh': test_dir / 'test.raw.zh'
}

# 复制原始文件到目标位置
for key in raw_files:
    src, dst = raw_files[key], target_files[key]
    if src.exists(): shutil.copy(src, dst)
print("✅ 原始文件重命名完成！")

# -----------------------------
# 文本清洗
# -----------------------------
def strQ2B(ustring):
    return ''.join([chr(32) if ord(c)==12288 else chr(ord(c)-65248) 
                    if 65281 <= ord(c) <= 65374 else c for c in ustring])

def clean_s(s, lang):
    if lang == 'en':
        s = re.sub(r"\([^()]*\)", "", s)
        s = s.replace('-', '')
        s = re.sub(r'([.,;!?()\"])', r' \1 ', s)
    elif lang == 'zh':
        s = strQ2B(s)
        s = re.sub(r"\([^()]*\)", "", s)
        s = s.replace(' ', '').replace('—', '').replace('_', '')
        s = s.replace('“', '"').replace('”', '"')
        s = re.sub(r'([。,;!?()\"~「」])', r' \1 ', s)
    return ' '.join(s.strip().split())

def len_s(s, lang):
    return len(s) if lang=='zh' else len(s.split())

def clean_corpus(prefix, l1, l2, ratio=9, max_len=1000, min_len=1):
    out_l1, out_l2 = Path(f'{prefix}.clean.{l1}'), Path(f'{prefix}.clean.{l2}')
    with open(f'{prefix}.{l1}', 'r', encoding='utf-8') as f1, \
         open(f'{prefix}.{l2}', 'r', encoding='utf-8') as f2, \
         open(out_l1, 'w', encoding='utf-8') as out1, \
         open(out_l2, 'w', encoding='utf-8') as out2:
        for s1 in f1:
            s1 = s1.strip()
            s2 = f2.readline().strip()
            s1, s2 = clean_s(s1, l1), clean_s(s2, l2)
            l1_len, l2_len = len_s(s1, l1), len_s(s2, l2)
            if min_len>0 and (l1_len<min_len or l2_len<min_len): continue
            if max_len>0 and (l1_len>max_len or l2_len>max_len): continue
            if ratio>0 and (l1_len/l2_len>ratio or l2_len/l1_len>ratio): continue
            print(s1, file=out1)
            print(s2, file=out2)
    return out_l1, out_l2

src_lang, tgt_lang = 'en', 'zh'
data_prefix = ted_dir / 'train_dev.raw'
test_prefix = test_dir / 'test.raw'

train_clean, valid_clean = clean_corpus(data_prefix, src_lang, tgt_lang)
test_clean = clean_corpus(test_prefix, src_lang, tgt_lang, ratio=-1, min_len=-1, max_len=-1)

# -----------------------------
# 划分 train / valid
# -----------------------------
def split_train_valid(prefix, valid_ratio=0.01):
    train_ratio = 1 - valid_ratio
    train_file = {lang: Path(f'{prefix.parent}/train.clean.{lang}') for lang in [src_lang, tgt_lang]}
    valid_file = {lang: Path(f'{prefix.parent}/valid.clean.{lang}') for lang in [src_lang, tgt_lang]}
    lines = {lang: open(f'{prefix}.clean.{lang}', 'r', encoding='utf-8').readlines() for lang in [src_lang, tgt_lang]}
    num_lines = len(lines[src_lang])
    indices = list(range(num_lines))
    random.shuffle(indices)
    cutoff = int(num_lines * train_ratio)
    for lang in [src_lang, tgt_lang]:
        with open(train_file[lang], 'w', encoding='utf-8') as f_train, \
             open(valid_file[lang], 'w', encoding='utf-8') as f_valid:
            for i, idx in enumerate(indices):
                (f_train if i<cutoff else f_valid).write(lines[lang][idx])
    return train_file, valid_file

train_file, valid_file = split_train_valid(data_prefix)

# -----------------------------
# SentencePiece 训练
# -----------------------------
vocab_size = 8000
spm_prefix = ted_dir / f'spm{vocab_size}'
spm_model_file = ted_dir / f'spm{vocab_size}.model'
if not spm_model_file.exists():
    print("训练 SentencePiece 模型中...")
    spm.SentencePieceTrainer.train(
        input=','.join([str(train_file[src_lang]), str(valid_file[src_lang]),
                        str(train_file[tgt_lang]), str(valid_file[tgt_lang])]),
        model_prefix=str(spm_prefix),
        vocab_size=vocab_size,
        character_coverage=1,
        model_type='unigram',
        input_sentence_size=int(1e6),
        shuffle_input_sentence=True,
        normalization_rule_name='nmt_nfkc_cf'
    )

spm_model = spm.SentencePieceProcessor(model_file=str(spm_model_file))

# -----------------------------
# SentencePiece 编码（覆盖旧文件）
# -----------------------------
def spm_encode_file(in_path, out_path, force=True):
    if out_path.exists() and not force:
        return
    with open(in_path, 'r', encoding='utf-8') as f_in, \
         open(out_path, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            tok = spm_model.encode(line.strip(), out_type=str)
            print(' '.join(tok), file=f_out)

for split in ['train', 'valid', 'test']:
    for lang in [src_lang, tgt_lang]:
        if split != 'test':
            in_path = ted_dir / f'{split}.clean.{lang}'
            out_path = ted_dir / f'{split}.{lang}'
        else:
            in_path = test_dir / f'test.raw.clean.{lang}'
            out_path = test_dir / f'{split}.{lang}'
        spm_encode_file(in_path, out_path, force=True)

# -----------------------------
# 导出 SPM 词表为 fairseq 可识别字典
# -----------------------------
dict_path = ted_dir / f'dict.txt'
with open(dict_path, 'w', encoding='utf-8') as f:
    for i in range(spm_model.get_piece_size()):
        piece = spm_model.id_to_piece(i)
        print(f"{piece} 1", file=f)

# -----------------------------
# 构建 binary 数据
# -----------------------------
from argparse import Namespace
from fairseq_cli import preprocess
from pathlib import Path

# binary 数据存放路径
binpath = ted_dir / 'data-bin'

# preprocess 需要的文件路径
trainpref = ted_dir / 'train'
validpref = ted_dir / 'valid'
testpref  = test_dir / 'test'

# # 清理旧字典，避免 FileExistsError
# for lang in ['en', 'zh']:
#     dict_file = binpath / f'dict.{lang}.txt'
#     if dict_file.exists():
#         dict_file.unlink()

binpath = ted_dir / 'data-bin'
if not binpath.exists():
    for lang in ['en','zh']:
        dict_file = binpath / f'dict.{lang}.txt'
        if dict_file.exists():
            dict_file.unlink()
    binpath.mkdir(parents=True, exist_ok=True)

    args = Namespace(
        task='translation',
        source_lang='en',
        target_lang='zh',
        trainpref=str(ted_dir / 'train'),
        validpref=str(ted_dir / 'valid'),
        testpref=str(test_dir / 'test'),
        destdir=str(binpath.resolve()),
        joined_dictionary=True,
        srcdict=None,
        tgtdict=None,
        vocab_file=str(ted_dir / 'vocab8000.model'),
        align_suffix=None,
        workers=1,
        only_source=False,
        padding_factor=1,
        thresholdsrc=0,
        thresholdtgt=0,
        nwordssrc=8000,
        nwordstgt=8000,
        dataset_impl='mmap',
        alignfile=None,
    )

    preprocess.main(args)
    print("✅ 数据预处理完成！")
else:
    print("✅ binary 数据已存在，跳过 preprocess")

config = Namespace(
    datadir = "./DATA/data-bin/ted2020",
    savedir = "./checkpoints/rnn",
    source_lang = "en",
    target_lang = "zh",
    
    # cpu threads when fetching & processing data.
    num_workers=2,  
    # batch size in terms of tokens. gradient accumulation increases the effective batchsize.
    max_tokens=8192,
    accum_steps=2,
    
    # the lr s calculated from Noam lr scheduler. you can tune the maximum lr by this factor.
    lr_factor=2.,
    lr_warmup=4000,
    
    # clipping gradient norm helps alleviate gradient exploding
    clip_norm=1.0,
    
    # maximum epochs for training
    max_epoch=30,
    start_epoch=1,
    
    # beam size for beam search
    beam=5, 
    # generate sequences of maximum length ax + b, where x is the source length
    max_len_a=1.2, 
    max_len_b=10,
    # when decoding, post process sentence by removing sentencepiece symbols.
    post_process = "sentencepiece",
    
    # checkpoints
    keep_last_epochs=5,
    resume=None, # if resume from checkpoint name (under config.savedir)
    
    # logging
    use_wandb=False,
)
from argparse import Namespace
from fairseq.tasks.translation import TranslationTask
import pprint
import logging
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level="INFO", # "DEBUG" "WARNING" "ERROR"
    stream=sys.stdout,
)
proj = "hw5.seq2seq"
logger = logging.getLogger(proj)
if config.use_wandb:
    import wandb
    wandb.init(project=proj, name=Path(config.savedir).stem, config=config)

from fairseq import utils
cuda_env = utils.CudaEnvironment()
utils.CudaEnvironment.pretty_print_cuda_env_list([cuda_env])
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# -----------------------------
# 构建 Namespace，适配 fairseq 0.10.x
# -----------------------------
task_args = Namespace(
    data=r'D:\zl\ml2021spring-hw5\ted2020\data-bin',                    # 数据路径
    source_lang=config.source_lang,         # 源语言
    target_lang=config.target_lang,         # 目标语言
    train_subset="train",                   # 训练集名称
    required_seq_len_multiple=8,            # 序列长度对齐
    dataset_impl="mmap",                     # 数据实现方式
    upsample_primary=1,                     # primary dataset upsample
    left_pad_source=True,                   # 源序列左对齐
    left_pad_target=False,                  # 目标序列左对齐
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
logger.info("loading data for epoch 1")
task.load_dataset(split="train", epoch=1, combine=True)  # combine if you have back-translation data
task.load_dataset(split="valid", epoch=1)

# -----------------------------
# 查看 sample
# -----------------------------
sample = task.dataset("valid")[1]

pprint.pprint(sample)

# 打印 source
source_str = task.source_dictionary.string(
    sample['source'],
    config.post_process,
)
pprint.pprint("Source: " + source_str)

# 打印 target
target_str = task.target_dictionary.string(
    sample['target'],
    config.post_process,
)
pprint.pprint("Target: " + target_str)


def load_data_iterator(task, split, epoch=1, max_tokens=4000, num_workers=1, cached=True):
    batch_iterator = task.get_batch_iterator(
        dataset=task.dataset(split),
        max_tokens=max_tokens,
        max_sentences=None,
        max_positions=utils.resolve_max_positions(
            task.max_positions(),
            max_tokens,
        ),
        ignore_invalid_inputs=True,
        seed=seed,
        num_workers=num_workers,
        epoch=epoch,
        disable_iterator_cache=not cached,
    )
    return batch_iterator
# -----------------------------
# Transformer 模型构建
# -----------------------------
import torch
import torch.nn as nn
from types import SimpleNamespace
from fairseq.models.transformer import TransformerEncoder, TransformerDecoder
from fairseq.models import FairseqEncoderDecoderModel

# =============================
# 定义模型参数
# =============================
model_args = SimpleNamespace(
    # 编码器参数
    encoder_embed_dim=512,
    encoder_layers=6,
    encoder_attention_heads=8,
    encoder_ffn_embed_dim=2048,
    encoder_normalize_before=False,

    # 解码器参数
    decoder_embed_dim=512,
    decoder_layers=6,
    decoder_attention_heads=8,
    decoder_ffn_embed_dim=2048,
    decoder_normalize_before=False,
    share_decoder_input_output_embed=False,

    # dropout 参数
    dropout=0.1,
    attention_dropout=0.1,
    activation_dropout=0.0,

    # 最大序列长度
    max_source_positions=1024,
    max_target_positions=1024,

    # 激活函数
    activation_fn='relu',
    layerdrop=0.0,
)

# =============================
# 构建 Transformer 模型函数
# =============================
def build_model(model_args):
    # -----------------------------
    # Encoder
    # -----------------------------
    encoder = TransformerEncoder(
        embed_dim=model_args.encoder_embed_dim,
        num_layers=model_args.encoder_layers,
        num_heads=model_args.encoder_attention_heads,
        ffn_embed_dim=model_args.encoder_ffn_embed_dim,
        dropout=model_args.dropout,
        attention_dropout=model_args.attention_dropout,
        activation_dropout=model_args.activation_dropout,
        max_source_positions=getattr(model_args, "max_source_positions", 1024),
        encoder_normalize_before=getattr(model_args, "encoder_normalize_before", False),
    )

    # -----------------------------
    # Decoder
    # -----------------------------
    decoder = TransformerDecoder(
        embed_dim=model_args.decoder_embed_dim,
        num_layers=model_args.decoder_layers,
        num_heads=model_args.decoder_attention_heads,
        ffn_embed_dim=model_args.decoder_ffn_embed_dim,
        dropout=model_args.dropout,
        attention_dropout=model_args.attention_dropout,
        activation_dropout=model_args.activation_dropout,
        max_target_positions=getattr(model_args, "max_target_positions", 1024),
        decoder_normalize_before=getattr(model_args, "decoder_normalize_before", False),
        share_decoder_input_output_embed=getattr(model_args, "share_decoder_input_output_embed", False),
    )

    # -----------------------------
    # 构建完整 Encoder-Decoder 模型
    # =============================
    model = FairseqEncoderDecoderModel(encoder, decoder)
    return model


class LabelSmoothedCrossEntropyCriterion(nn.Module):
    def __init__(self, smoothing, ignore_index=None, reduce=True):
        super().__init__()
        self.smoothing = smoothing
        self.ignore_index = ignore_index
        self.reduce = reduce
    
    def forward(self, lprobs, target):
        if target.dim() == lprobs.dim() - 1:
            target = target.unsqueeze(-1)
        # nll: Negative log likelihood，當目標是one-hot時的cross-entropy loss. 以下同 F.nll_loss
        nll_loss = -lprobs.gather(dim=-1, index=target)
        # 將一部分正確答案的機率分配給其他label 所以當計算cross-entropy時等於把所有label的log prob加起來
        smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
        if self.ignore_index is not None:
            pad_mask = target.eq(self.ignore_index)
            nll_loss.masked_fill_(pad_mask, 0.0)
            smooth_loss.masked_fill_(pad_mask, 0.0)
        else:
            nll_loss = nll_loss.squeeze(-1)
            smooth_loss = smooth_loss.squeeze(-1)
        if self.reduce:
            nll_loss = nll_loss.sum()
            smooth_loss = smooth_loss.sum()
        # 計算cross-entropy時 加入分配給其他label的loss
        eps_i = self.smoothing / lprobs.size(-1)
        loss = (1.0 - self.smoothing) * nll_loss + eps_i * smooth_loss
        return loss
class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
    
    @property
    def param_groups(self):
        return self.optimizer.param_groups
        
    def multiply_grads(self, c):
        """Multiplies grads by a constant *c*."""                
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    p.grad.data.mul_(c)
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return 0 if not step else self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))

from fairseq.data import iterators
from torch.cuda.amp import GradScaler, autocast
from tqdm.auto import tqdm
def train_one_epoch(epoch_itr, model, task, criterion, optimizer, accum_steps=1):
    itr = epoch_itr.next_epoch_itr(shuffle=True)
    itr = iterators.GroupedIterator(itr, accum_steps) # 梯度累積: 每 accum_steps 個 sample 更新一次
    
    stats = {"loss": []}
    scaler = GradScaler() # 混和精度訓練 automatic mixed precision (amp) 
    
    model.train()
    progress = tqdm.tqdm(itr, desc=f"train epoch {epoch_itr.epoch}", leave=False)
    for samples in progress:
        model.zero_grad()
        accum_loss = 0
        sample_size = 0
        # 梯度累積: 每 accum_steps 個 sample 更新一次
        for i, sample in enumerate(samples):
            if i == 1:
                # emptying the CUDA cache after the first step can reduce the chance of OOM
                torch.cuda.empty_cache()

            sample = utils.move_to_cuda(sample, device=device)
            target = sample["target"]
            sample_size_i = sample["ntokens"]
            sample_size += sample_size_i
            
            # 混和精度訓練 
            with autocast():
                net_output = model.forward(**sample["net_input"])
                lprobs = F.log_softmax(net_output[0], -1)            
                loss = criterion(lprobs.view(-1, lprobs.size(-1)), target.view(-1))
                
                # logging
                accum_loss += loss.item()
                # back-prop
                scaler.scale(loss).backward()                
        
        scaler.unscale_(optimizer)
        optimizer.multiply_grads(1 / (sample_size or 1.0)) # (sample_size or 1.0) handles the case of a zero gradient
        gnorm = nn.utils.clip_grad_norm_(model.parameters(), config.clip_norm) # 梯度裁剪 防止梯度爆炸
        
        scaler.step(optimizer)
        scaler.update()
        
        # logging
        loss_print = accum_loss/sample_size
        stats["loss"].append(loss_print)
        progress.set_postfix(loss=loss_print)
        if config.use_wandb:
            wandb.log({
                "train/loss": loss_print,
                "train/grad_norm": gnorm.item(),
                "train/lr": optimizer.rate(),
                "train/sample_size": sample_size,
            })
        
    loss_print = np.mean(stats["loss"])
    logger.info(f"training loss: {loss_print:.4f}")
    return stats

model = build_model(model_args)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model.to(device)

sequence_generator = task.build_generator([model], config)

def decode(toks, dictionary):
    # 從 Tensor 轉成人看得懂的句子
    s = dictionary.string(
        toks.int().cpu(),
        config.post_process,
    )
    return s if s else "<unk>"

def inference_step(sample, model):
    gen_out = sequence_generator.generate([model], sample)
    srcs = []
    hyps = []
    refs = []
    for i in range(len(gen_out)):
        # 對於每個 sample, 收集輸入，輸出和參考答案，稍後計算 BLEU
        srcs.append(decode(
            utils.strip_pad(sample["net_input"]["src_tokens"][i], task.source_dictionary.pad()), 
            task.source_dictionary,
        ))
        hyps.append(decode(
            gen_out[i][0]["tokens"], # 0 代表取出 beam 內分數第一的輸出結果
            task.target_dictionary,
        ))
        refs.append(decode(
            utils.strip_pad(sample["target"][i], task.target_dictionary.pad()), 
            task.target_dictionary,
        ))
    return srcs, hyps, refs
import shutil
import sacrebleu

def validate(model, task, criterion, log_to_wandb=True):
    logger.info('begin validation')
    itr = load_data_iterator(task, "valid", 1, config.max_tokens, config.num_workers).next_epoch_itr(shuffle=False)
    
    stats = {"loss":[], "bleu": 0, "srcs":[], "hyps":[], "refs":[]}
    srcs = []
    hyps = []
    refs = []
    
    model.eval()
    progress = tqdm.tqdm(itr, desc=f"validation", leave=False)
    with torch.no_grad():
        for i, sample in enumerate(progress):
            # validation loss
            sample = utils.move_to_cuda(sample, device=device)
            net_output = model.forward(**sample["net_input"])

            lprobs = F.log_softmax(net_output[0], -1)
            target = sample["target"]
            sample_size = sample["ntokens"]
            loss = criterion(lprobs.view(-1, lprobs.size(-1)), target.view(-1)) / sample_size
            progress.set_postfix(valid_loss=loss.item())
            stats["loss"].append(loss)
            
            # 進行推論
            s, h, r = inference_step(sample, model)
            srcs.extend(s)
            hyps.extend(h)
            refs.extend(r)
            
    tok = 'zh' if task.cfg.target_lang == 'zh' else '13a'
    stats["loss"] = torch.stack(stats["loss"]).mean().item()
    stats["bleu"] = sacrebleu.corpus_bleu(hyps, [refs], tokenize=tok) # 計算BLEU score
    stats["srcs"] = srcs
    stats["hyps"] = hyps
    stats["refs"] = refs
    
    if config.use_wandb and log_to_wandb:
        wandb.log({
            "valid/loss": stats["loss"],
            "valid/bleu": stats["bleu"].score,
        }, commit=False)
    
    showid = np.random.randint(len(hyps))
    logger.info("example source: " + srcs[showid])
    logger.info("example hypothesis: " + hyps[showid])
    logger.info("example reference: " + refs[showid])
    
    # show bleu results
    logger.info(f"validation loss:\t{stats['loss']:.4f}")
    logger.info(stats["bleu"].format())
    return stats

def validate_and_save(model, task, criterion, optimizer, epoch, save=True):   
    stats = validate(model, task, criterion)
    bleu = stats['bleu']
    loss = stats['loss']
    if save:
        # save epoch checkpoints
        savedir = Path(config.savedir).absolute()
        savedir.mkdir(parents=True, exist_ok=True)
        
        check = {
            "model": model.state_dict(),
            "stats": {"bleu": bleu.score, "loss": loss},
            "optim": {"step": optimizer._step}
        }
        torch.save(check, savedir/f"checkpoint{epoch}.pt")
        shutil.copy(savedir/f"checkpoint{epoch}.pt", savedir/f"checkpoint_last.pt")
        logger.info(f"saved epoch checkpoint: {savedir}/checkpoint{epoch}.pt")
    
        # save epoch samples
        with open(savedir/f"samples{epoch}.{config.source_lang}-{config.target_lang}.txt", "w") as f:
            for s, h in zip(stats["srcs"], stats["hyps"]):
                f.write(f"{s}\t{h}\n")

        # get best valid bleu    
        if getattr(validate_and_save, "best_bleu", 0) < bleu.score:
            validate_and_save.best_bleu = bleu.score
            torch.save(check, savedir/f"checkpoint_best.pt")
            
        del_file = savedir / f"checkpoint{epoch - config.keep_last_epochs}.pt"
        if del_file.exists():
            del_file.unlink()
    
    return stats

def try_load_checkpoint(model, optimizer=None, name=None):
    name = name if name else "checkpoint_last.pt"
    checkpath = Path(config.savedir)/name
    if checkpath.exists():
        check = torch.load(checkpath)
        model.load_state_dict(check["model"])
        stats = check["stats"]
        step = "unknown"
        if optimizer != None:
            optimizer._step = step = check["optim"]["step"]
        logger.info(f"loaded checkpoint {checkpath}: step={step} loss={stats['loss']} bleu={stats['bleu']}")
    else:
        logger.info(f"no checkpoints found at {checkpath}!")

if __name__ == "__main__":
    import pprint

    # 初始化任务
    task = TranslationTask.setup_task(task_args)

    # 加载数据集
    logger.info("loading data for epoch 1")
    task.load_dataset(split="train", epoch=1)
    task.load_dataset(split="valid", epoch=1)

    # 查看 sample
    sample = task.dataset("valid")[1]
    pprint.pprint(sample)

    # 打印 source/target
    source_str = task.source_dictionary.string(sample['source'], config.post_process)
    target_str = task.target_dictionary.string(sample['target'], config.post_process)
    pprint.pprint("Source: " + source_str)
    pprint.pprint("Target: " + target_str)


    # 创建 valid iterator demo
    demo_epoch_obj = load_data_iterator(task, "valid", epoch=1, max_tokens=80, num_workers=0, cached=True)
    demo_iter = demo_epoch_obj.next_epoch_itr(shuffle=True)
    batch_sample = next(demo_iter)

    print("\n✅ Demo batch sample keys:", batch_sample.keys())
    # decode source/target tokens
    decoded_sources = [task.source_dictionary.string(s, 'sentencepiece') for s in batch_sample['net_input']['src_tokens']]
    decoded_targets = [task.target_dictionary.string(t, 'sentencepiece') for t in batch_sample['target']]
    print("\nDecoded source:", decoded_sources)
    print("Decoded target:", decoded_targets)
    # 一般都用0.1效果就很好了
    criterion = LabelSmoothedCrossEntropyCriterion(
    smoothing=0.1,
    ignore_index=task.target_dictionary.pad(),
    )
    optimizer = NoamOpt(
    model_size=model_args.encoder_embed_dim, 
    factor=config.lr_factor, 
    warmup=config.lr_warmup, 
    optimizer=torch.optim.AdamW(
        model.parameters(), 
        lr=0, betas=(0.9, 0.98), eps=1e-9, weight_decay=0.0001
    )
    )
    criterion = criterion.to(device=device)
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
    logger.info(f"max tokens per batch = {config.max_tokens}, accumulate steps = {config.accum_steps}")
    epoch_itr = load_data_iterator(task, "train", config.start_epoch, config.max_tokens, config.num_workers)
    try_load_checkpoint(model, optimizer, name=config.resume)
    while epoch_itr.next_epoch_idx <= config.max_epoch:
        # train for one epoch
        train_one_epoch(epoch_itr, model, task, criterion, optimizer, config.accum_steps)
        stats = validate_and_save(model, task, criterion, optimizer, epoch=epoch_itr.epoch)
        logger.info("end of epoch {}".format(epoch_itr.epoch))    
        epoch_itr = load_data_iterator(task, "train", epoch_itr.next_epoch_idx, config.max_tokens, config.num_workers)
    import tqdm
    import torch
    from fairseq import utils

    # -----------------------------
    # 1️⃣ 加载 checkpoint 并验证
    # -----------------------------
    try_load_checkpoint(model, name="avg_last_5_checkpoint.pt")
    validate(model, task, criterion, log_to_wandb=False)

    # -----------------------------
    # 2️⃣ 定义预测函数
    # -----------------------------
    def generate_prediction(model, task, split="test", outfile="./submission.txt", post_process=True):
        """
        使用模型对指定数据集 split 进行预测，并保存到 outfile。
        post_process=True 会自动去掉 sentencepiece 的 "▁" 和 <unk>。
        """
        # 加载数据集
        task.load_dataset(split=split, epoch=1)
        
        # 创建迭代器
        itr = load_data_iterator(task, split, 1, config.max_tokens, config.num_workers).next_epoch_itr(shuffle=False)
        
        idxs = []
        hyps = []

        model.eval()
        progress = tqdm.tqdm(itr, desc=f"Prediction ({split})")
        with torch.no_grad():
            for sample in progress:
                sample = utils.move_to_cuda(sample, device=device)
                
                # 推理
                _, batch_hyps, _ = inference_step(sample, model)
                
                hyps.extend(batch_hyps)
                idxs.extend(list(sample['id']))

        # 根据 preprocess 时的顺序排列
        hyps = [x for _, x in sorted(zip(idxs, hyps))]

        # 可选后处理：去掉 sentencepiece 标记和 <unk>
        if post_process:
            hyps = [h.replace("▁", " ").replace("<unk>", "").strip() for h in hyps]

        # 保存到文件
        with open(outfile, "w", encoding="utf-8") as f:
            for h in hyps:
                f.write(h + "\n")

        print(f"✅ Prediction saved to {outfile}, total {len(hyps)} lines.")

    # -----------------------------
    # 3️⃣ 调用预测函数
    # -----------------------------
    generate_prediction(model, task, split="test", outfile="./submission.txt")
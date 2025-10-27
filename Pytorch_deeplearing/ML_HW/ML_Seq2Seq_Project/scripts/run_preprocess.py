import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import random
import numpy as np
import torch
import shutil
import sentencepiece as spm

from configs.config import SEED, TED_DIR, TEST_DIR, RAW_FILES, TARGET_FILES, VOCAB_SIZE, SPM_MODEL_PREFIX, SPM_MODEL_FILE
from configs.config import DATA_BIN_DIR, TRAIN_RATIO
from preprocess.clean import clean_corpus, split_train_valid
from preprocess.spm_encode import train_spm, encode_file

# -----------------------------
# 随机种子
# -----------------------------
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# -----------------------------
# 原始文件复制
# -----------------------------
for key in RAW_FILES:
    src, dst = RAW_FILES[key], TARGET_FILES[key]
    if dst.exists():
        print(f"⚡ {dst.name} 已存在，跳过复制")
        continue
    if src.exists():
        shutil.copy(src, dst)
        print(f"✅ 复制 {src.name} 到 {dst.name}")
print("✅ 原始文件处理完成！")

# -----------------------------
# 数据清洗
# -----------------------------
train_prefix = TED_DIR / "train_dev.raw"
test_prefix = TEST_DIR / "test.raw"

# 先检查是否已经有 clean 文件
train_clean_files_exist = all((TED_DIR / f'{lang}.clean').exists() for lang in ['en', 'zh'])
if not train_clean_files_exist:
    train_clean, valid_clean = clean_corpus(train_prefix, 'en', 'zh')
    train_file, valid_file = split_train_valid(train_prefix, 'en', 'zh')
    test_clean = clean_corpus(test_prefix, 'en', 'zh', ratio=-1, min_len=-1, max_len=-1)
    print("✅ 数据清洗完成！")
else:
    print("⚡ clean 文件已存在，跳过数据清洗")
    train_file = {'en': TED_DIR / 'train.clean.en', 'zh': TED_DIR / 'train.clean.zh'}
    valid_file = {'en': TED_DIR / 'valid.clean.en', 'zh': TED_DIR / 'valid.clean.zh'}

# -----------------------------
# SentencePiece
# -----------------------------
if not SPM_MODEL_FILE.exists():
    spm_model_file = train_spm(
        [train_file['en'], valid_file['en'], train_file['zh'], valid_file['zh']],
        SPM_MODEL_PREFIX,
        vocab_size=VOCAB_SIZE
    )
    print("✅ SentencePiece 训练完成！")
else:
    spm_model_file = SPM_MODEL_FILE
    print(f"⚡ {SPM_MODEL_FILE.name} 已存在，跳过训练")

spm_model = spm.SentencePieceProcessor(model_file=str(spm_model_file))

for split in ['train', 'valid', 'test']:
    for lang in ['en', 'zh']:
        if split != 'test':
            in_path = TED_DIR / f'{split}.clean.{lang}'
            out_path = TED_DIR / f'{split}.{lang}'
        else:
            in_path = TEST_DIR / f'test.raw.clean.{lang}'
            out_path = TEST_DIR / f'{split}.{lang}'
        if out_path.exists():
            print(f"⚡ {out_path.name} 已存在，跳过编码")
            continue
        encode_file(spm_model, in_path, out_path, force=True)
print("✅ SentencePiece 编码完成！")

# -----------------------------
# fairseq data-bin
# -----------------------------
from argparse import Namespace
from fairseq_cli import preprocess

if any(DATA_BIN_DIR.glob('*')):
    print(f"⚡ {DATA_BIN_DIR} 已存在数据，跳过 data-bin 构建")
else:
    DATA_BIN_DIR.mkdir(exist_ok=True, parents=True)
    args  = Namespace(
    task='translation',
    source_lang='en',
    target_lang='zh',
    trainpref=str(TED_DIR / 'train'),
    validpref=str(TED_DIR / 'valid'),
    testpref=str(TEST_DIR / 'test'),
    destdir=str(DATA_BIN_DIR),
    joined_dictionary=True,
    workers=1,
    only_source=False,
    padding_factor=1,
    nwordssrc=VOCAB_SIZE,
    nwordstgt=VOCAB_SIZE,
    dataset_impl='mmap',
    srcdict=None,
    tgtdict=None,
    thresholdsrc=0,       # ✅ 必须加
    thresholdtgt=0,       # ✅ 必须加
    alignfile=None,       # ✅ 必须加
    align_suffix=None     # ✅ 必须加
)

    preprocess.main(args)
    print("✅ fairseq 数据二进制化完成！")

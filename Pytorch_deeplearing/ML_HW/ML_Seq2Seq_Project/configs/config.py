from pathlib import Path

# -----------------------------
# 数据路径
# -----------------------------
DATA_ROOT = Path(r"D:\zl\ml2021spring-hw5")
DATASET_NAME = "ted2020"

TED_DIR = DATA_ROOT / DATASET_NAME
TEST_DIR = DATA_ROOT / "test"

RAW_FILES = {
    'train_en': TED_DIR / 'raw.en',
    'train_zh': TED_DIR / 'raw.zh',
    'test_en': TEST_DIR / 'test.en',
    'test_zh': TEST_DIR / 'test.zh'
}

TARGET_FILES = {
    'train_en': TED_DIR / 'train_dev.raw.en',
    'train_zh': TED_DIR / 'train_dev.raw.zh',
    'test_en': TEST_DIR / 'test.raw.en',
    'test_zh': TEST_DIR / 'test.raw.zh'
}

# -----------------------------
# 训练参数
# -----------------------------
SEED = 73
VOCAB_SIZE = 8000
SPM_MODEL_PREFIX = TED_DIR / f'spm{VOCAB_SIZE}'
SPM_MODEL_FILE = TED_DIR / f'spm{VOCAB_SIZE}.model'

TRAIN_RATIO = 0.99
# -----------------------------
# fairseq 数据二进制化
# -----------------------------
DATA_BIN_DIR = TED_DIR / "data-bin"
# -----------------------------
# Training / runtime parameters
# -----------------------------
max_tokens = 4000
accum_steps = 2
num_workers = 0
start_epoch = 1
max_epoch = 30
resume = None  # 如果想从 checkpoint 恢复，可以写 "checkpoint_last.pt"
savedir = "./checkpoints"
keep_last_epochs = 5
use_wandb = False
clip_norm = 1.0

# source / target languages
source_lang = 'en'
target_lang = 'zh'
# 新增 clip_norm
clip_norm = 1.0
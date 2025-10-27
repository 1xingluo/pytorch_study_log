import re
from pathlib import Path
import random

# -----------------------------
# 字符全角转半角
# -----------------------------
def strQ2B(ustring):
    return ''.join([chr(32) if ord(c)==12288 else chr(ord(c)-65248) 
                    if 65281 <= ord(c) <= 65374 else c for c in ustring])

# -----------------------------
# 文本清洗
# -----------------------------
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

# -----------------------------
# 清洗语料
# -----------------------------
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

# -----------------------------
# 划分 train / valid
# -----------------------------
def split_train_valid(prefix, l1, l2, valid_ratio=0.01):
    train_ratio = 1 - valid_ratio
    train_file = {lang: Path(f'{prefix.parent}/train.clean.{lang}') for lang in [l1, l2]}
    valid_file = {lang: Path(f'{prefix.parent}/valid.clean.{lang}') for lang in [l1, l2]}
    lines = {lang: open(f'{prefix}.clean.{lang}', 'r', encoding='utf-8').readlines() for lang in [l1, l2]}
    num_lines = len(lines[l1])
    indices = list(range(num_lines))
    random.shuffle(indices)
    cutoff = int(num_lines * train_ratio)
    for lang in [l1, l2]:
        with open(train_file[lang], 'w', encoding='utf-8') as f_train, \
             open(valid_file[lang], 'w', encoding='utf-8') as f_valid:
            for i, idx in enumerate(indices):
                (f_train if i<cutoff else f_valid).write(lines[lang][idx])
    return train_file, valid_file

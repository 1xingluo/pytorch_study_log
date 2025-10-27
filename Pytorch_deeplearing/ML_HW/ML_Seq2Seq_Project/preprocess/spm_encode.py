import sentencepiece as spm
from pathlib import Path

def train_spm(train_files, model_prefix, vocab_size=8000):
    spm.SentencePieceTrainer.train(
        input=','.join([str(f) for f in train_files]),
        model_prefix=str(model_prefix),
        vocab_size=vocab_size,
        character_coverage=1,
        model_type='unigram',
        input_sentence_size=int(1e6),
        shuffle_input_sentence=True,
        normalization_rule_name='nmt_nfkc_cf'
    )
    return Path(str(model_prefix)+'.model')

def encode_file(spm_model, in_path, out_path, force=True):
    if out_path.exists() and not force:
        return
    with open(in_path, 'r', encoding='utf-8') as f_in, \
         open(out_path, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            tok = spm_model.encode(line.strip(), out_type=str)
            print(' '.join(tok), file=f_out)

import torch
import tqdm
from pathlib import Path
import sys

# -----------------------------
# 将项目根目录加入 sys.path
# -----------------------------
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# -----------------------------
# 引入模块
# -----------------------------
from checkpoint_utils import try_load_checkpoint  # 你的本地 checkpoint_utils.py
from validate import validate                     # 你的 validate 函数
from run_train import load_data_iterator, task, model, criterion, utils, device

# -----------------------------
# 本地 config 定义
# -----------------------------
class Config:
    max_tokens = 4000
    num_workers = 0
    savedir = "./checkpoints"
    source_lang = 'en'
    target_lang = 'zh'

config = Config()

# -----------------------------
# 加载最后一次 checkpoint
# -----------------------------
print("📌 Loading best checkpoint...")
try_load_checkpoint(model, name="checkpoint_best.pt")

# -----------------------------
# 验证模型
# -----------------------------
print("📌 Running validation...")
validate(model, task, criterion, logger=None, device=device, utils=utils,
         load_data_iterator_fn=load_data_iterator, max_tokens=config.max_tokens,
         num_workers=config.num_workers, target_lang=config.target_lang, use_wandb=False)



from argparse import Namespace

# -----------------------------
# 构建 sequence generator
# -----------------------------
gen_args = Namespace(beam=5, max_len_a=1.2, max_len_b=10, stop_early=True)
sequence_generator = task.build_generator([model], gen_args)

# -----------------------------
# decode 函数
# -----------------------------
def decode(tensor, dictionary):
    return dictionary.string(tensor.int().cpu()).replace("@@ ", "")

# -----------------------------
# 自定义 inference_step
# -----------------------------
def inference_step(sample, model):
    gen_out = sequence_generator.generate([model], sample)
    srcs = []
    hyps = []
    refs = []
    for i in range(len(gen_out)):
        srcs.append(decode(
            utils.strip_pad(sample["net_input"]["src_tokens"][i], task.source_dictionary.pad()), 
            task.source_dictionary,
        ))
        hyps.append(decode(
            gen_out[i][0]["tokens"],  # 取 beam 第一的输出
            task.target_dictionary,
        ))
        refs.append(decode(
            utils.strip_pad(sample["target"][i], task.target_dictionary.pad()), 
            task.target_dictionary,
        ))
    return srcs, hyps, refs

# -----------------------------
# 生成预测
# -----------------------------
def generate_prediction(model, task, split="test", outfile="./prediction.txt"):
    print(f"📌 Generating predictions for split '{split}'...")
    task.load_dataset(split=split, epoch=1)
    itr = load_data_iterator(task, split, 1, config.max_tokens, config.num_workers).next_epoch_itr(shuffle=False)

    idxs = []
    hyps = []

    model.eval()
    progress = tqdm.tqdm(itr, desc=f"prediction")
    with torch.no_grad():
        for i, sample in enumerate(progress):
            sample = utils.move_to_cuda(sample, device=device)

            # 进行推理
            s, h, r = inference_step(sample, model)  # 注意你要保证 inference_step 已导入
            hyps.extend(h)
            idxs.extend(list(sample['id']))

    # 按原顺序排列
    hyps = [x for _, x in sorted(zip(idxs, hyps))]

    with open(outfile, "w", encoding="utf-8") as f:
        for h in hyps:
            f.write(h + "\n")

    print(f"✅ Prediction done! Results saved to {outfile}")


# -----------------------------
# 执行预测
# -----------------------------
generate_prediction(model, task)

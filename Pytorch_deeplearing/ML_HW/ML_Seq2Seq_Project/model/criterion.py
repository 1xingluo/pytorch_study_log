# model/criterion.py
import torch
import torch.nn as nn

class LabelSmoothedCrossEntropyCriterion(nn.Module):
    """
    带 label smoothing 的交叉熵损失
    """
    def __init__(self, smoothing=0.1, ignore_index=None, reduce=True):
        super().__init__()
        self.smoothing = smoothing
        self.ignore_index = ignore_index
        self.reduce = reduce
    
    def forward(self, lprobs, target):
        """
        lprobs: log probabilities, shape [batch, seq_len, vocab_size]
        target: target indices, shape [batch, seq_len]
        """
        if target.dim() == lprobs.dim() - 1:
            target = target.unsqueeze(-1)
        
        # Negative log likelihood
        nll_loss = -lprobs.gather(dim=-1, index=target)
        # smooth loss: 平均分配到所有 label
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

        eps_i = self.smoothing / lprobs.size(-1)
        loss = (1.0 - self.smoothing) * nll_loss + eps_i * smooth_loss
        return loss


# -----------------------------
# 示例：如何在训练中实例化
# -----------------------------
# 假设 task 是你 run_preprocess/run_train 初始化的 TranslationTask
# criterion = LabelSmoothedCrossEntropyCriterion(
#     smoothing=0.1,
#     ignore_index=task.target_dictionary.pad(),
# )

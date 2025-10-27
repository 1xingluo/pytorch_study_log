# model/optim.py
class NoamOpt:
    """
    Optimizer wrapper that implements the Noam learning rate schedule
    as described in the Transformer paper:
    lrate = factor * model_size ** (-0.5) * min(step_num ** (-0.5), step_num * warmup_steps ** (-1.5))
    """
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
        """Multiply gradients by a constant c."""
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    p.grad.data.mul_(c)

    def step(self):
        """Update parameters and learning rate"""
        self._step += 1
        rate = self.rate()
        for group in self.param_groups:
            group['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        """Compute learning rate at current step"""
        if step is None:
            step = self._step
        if step == 0:
            return 0
        return self.factor * (
            self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5))
        )

# -----------------------------
# 使用示例：
# -----------------------------
# import torch.optim as optim
# base_optimizer = optim.Adam(model.parameters(), lr=0)
# optimizer = NoamOpt(model_size=512, factor=2, warmup=4000, optimizer=base_optimizer)

import random
import torch
import numpy as np

def same_seeds(seed):
    # Python built-in random module
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
class Config:
    def __init__(self):
        # ---- 基本设置 ----
        self.seed=23442
        self.batch_size = 5000
        self.learning_rate = 1e-3
        self.num_epochs = 50
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_type = 'cnn'# selecting a model type from {'cnn', 'fcn', 'vae', 'resnet'}
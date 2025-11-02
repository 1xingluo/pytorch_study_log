from transformers import AdamW, BertForQuestionAnswering, BertTokenizerFast
import torch
class Config:
    def __init__(self):
        # ---- 基本设置 ----
        self.seed=0
        self.fp16_training =True
        self.train_batch_size = 16
        self.validation = True
        self.logging_step = 50
        self.learning_rate = 1e-4
        self.num_epoch = 2
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_path='C:\\Users\\xingluo\\Downloads'
        self.model =  BertForQuestionAnswering.from_pretrained(self.model_path).to(self.device)
        self.tokenizer = BertTokenizerFast.from_pretrained(self.model_path)
        # ---- 加速与设备设置 ----
        if self.fp16_training:
            from accelerate import Accelerator
            self.accelerator = Accelerator(fp16=True)
            self.device = self.accelerator.device
        else:
            self.accelerator = None
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

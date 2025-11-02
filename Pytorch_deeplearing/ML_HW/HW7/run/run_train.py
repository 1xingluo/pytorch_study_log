import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dataset.predata import cfg
from dataset.QA_Dataset import train_loader
from transformers import AdamW
from seed import same_seeds
same_seeds(cfg.seed)
device=cfg.device
num_epoch = cfg.num_epoch 
validation = cfg.validation
logging_step = cfg.logging_step
model=cfg.model
learning_rate = cfg.learning_rate
optimizer = AdamW(model.parameters(), lr=learning_rate)
if cfg.fp16_training:
    model, optimizer, train_loader = cfg.accelerator.prepare(cfg.model, optimizer, train_loader) 


model.train()

print("Start Training ...")
from tqdm.auto import tqdm
import torch
for epoch in range(num_epoch):
    step = 1
    train_loss = train_acc = 0
    for data in tqdm(train_loader):	
        # Load all data into GPU
        data = [i.to(device) for i in data]
        
        # Model inputs: input_ids, token_type_ids, attention_mask, start_positions, end_positions (Note: only "input_ids" is mandatory)
        # Model outputs: start_logits, end_logits, loss (return when start_positions/end_positions are provided)  
        output = model(input_ids=data[0], token_type_ids=data[1], attention_mask=data[2], start_positions=data[3], end_positions=data[4])

        # Choose the most probable start position / end position
        start_index = torch.argmax(output.start_logits, dim=1)
        end_index = torch.argmax(output.end_logits, dim=1)
        
        # Prediction is correct only if both start_index and end_index are correct
        train_acc += ((start_index == data[3]) & (end_index == data[4])).float().mean()
        train_loss += output.loss
        
        if cfg.fp16_training:
            cfg.accelerator.backward(output.loss)
        else:
            output.loss.backward()
        
        optimizer.step()
        optimizer.zero_grad()
        step += 1
        ##### ✅ Apply linear learning rate decay #####
        # 全局总步数 = 每个 epoch 的步数 × 总 epoch 数
        total_steps = len(train_loader) * num_epoch
        current_step = epoch * len(train_loader) + step
        # 线性下降 lr：lr = initial_lr * (1 - current_step / total_steps)
        lr_now = learning_rate * (1 - current_step / total_steps)
        for param_group in optimizer.param_groups:
            param_group["lr"] = max(lr_now, 1e-7)  # 防止过低
     
        # Print training loss and accuracy over past logging step
        if step % logging_step == 0:
            print(f"Epoch {epoch + 1} | Step {step} | loss = {train_loss.item() / logging_step:.3f}, acc = {train_acc / logging_step:.3f}")
            print(f"LR = {lr_now:.8f}")
            train_loss = train_acc = 0
    from dataset.QA_Dataset import dev_loader,dev_questions
    from evaluate import evaluate
    if validation:
        print("Evaluating Dev Set ...")
        model.eval()
        with torch.no_grad():
            dev_acc = 0
            for i, data in enumerate(tqdm(dev_loader)):
                output = model(input_ids=data[0].squeeze(dim=0).to(device), token_type_ids=data[1].squeeze(dim=0).to(device),
                       attention_mask=data[2].squeeze(dim=0).to(device))
                # prediction is correct only if answer text exactly matches
                dev_acc += evaluate(data, output) == dev_questions[i]["answer_text"]
            print(f"Validation | Epoch {epoch + 1} | acc = {dev_acc / len(dev_loader):.3f}")
        model.train()

# Save a model and its configuration file to the directory 「saved_model」 
# i.e. there are two files under the direcory 「saved_model」: 「pytorch_model.bin」 and 「config.json」
# Saved model can be re-loaded using 「model = BertForQuestionAnswering.from_pretrained("saved_model")」
import torch 
from evaluate import evaluate
from tqdm.auto import tqdm
print("Saving Model ...")
model_save_dir = "saved_model" 
model.save_pretrained(model_save_dir)




print("Evaluating Test Set ...")
result = []
from dataset.QA_Dataset import test_loader,test_questions
model.eval()
with torch.no_grad():
    for data in tqdm(test_loader):
        output = model(input_ids=data[0].squeeze(dim=0).to(device), token_type_ids=data[1].squeeze(dim=0).to(device),
                       attention_mask=data[2].squeeze(dim=0).to(device))
        result.append(evaluate(data, output))

result_file = "result.csv"
with open(result_file, 'w', encoding='utf-8') as f:	
    f.write("ID,Answer\n")
    for i, test_question in enumerate(test_questions):
        # 防止 result[i] 不是字符串
        answer = str(result[i]).replace(',', '') if result[i] else ""
        f.write(f"{test_question['id']},{answer}\n")

print(f"✅ Completed! Result saved to {os.path.abspath(result_file)}")
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from Config.config import Config
cfg=Config()
def read_data(file):
    with open(file, 'r', encoding="utf-8") as reader:
        data = json.load(reader)
    return data["questions"], data["paragraphs"]

train_questions, train_paragraphs = read_data("D:/zl/ml2021spring-hw7/ml2021-spring-hw7/hw7_train.json")
dev_questions, dev_paragraphs = read_data("D:/zl/ml2021spring-hw7/ml2021-spring-hw7/hw7_dev.json")
test_questions, test_paragraphs = read_data("D:/zl/ml2021spring-hw7/ml2021-spring-hw7/hw7_test.json")
# print(train_questions[0])
# print(train_paragraphs[3884])
train_questions_tokenized = cfg.tokenizer([train_question["question_text"] for train_question in train_questions], add_special_tokens=False)
dev_questions_tokenized = cfg.tokenizer([dev_question["question_text"] for dev_question in dev_questions], add_special_tokens=False)
test_questions_tokenized = cfg.tokenizer([test_question["question_text"] for test_question in test_questions], add_special_tokens=False) 


train_paragraphs_tokenized =cfg.tokenizer(train_paragraphs, add_special_tokens=False)
dev_paragraphs_tokenized = cfg.tokenizer(dev_paragraphs, add_special_tokens=False)
test_paragraphs_tokenized = cfg.tokenizer(test_paragraphs, add_special_tokens=False)

print(train_questions_tokenized[0])
print(train_paragraphs_tokenized[3884].char_to_token(141))
print(train_paragraphs_tokenized[3884].tokens[139:143])
# You can safely ignore the warning message as tokenized sequences will be futher processed in datset __getitem__ before passing to model
     
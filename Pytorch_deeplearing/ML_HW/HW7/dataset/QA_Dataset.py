import torch
from torch.utils.data import DataLoader, Dataset
import random
class QA_Dataset(Dataset):
    def __init__(self, split, questions, tokenized_questions, tokenized_paragraphs):
        self.split = split
        self.questions = questions
        self.tokenized_questions = tokenized_questions
        self.tokenized_paragraphs = tokenized_paragraphs
        self.max_question_len = 40
        self.max_paragraph_len = 320
        
        ##### TODO: Change value of doc_stride #####
        self.doc_stride = 50

        # Input sequence length = [CLS] + question + [SEP] + paragraph + [SEP]
        self.max_seq_len = 1 + self.max_question_len + 1 + self.max_paragraph_len + 1

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        question = self.questions[idx]
        tokenized_question = self.tokenized_questions[idx]
        tokenized_paragraph = self.tokenized_paragraphs[question["paragraph_id"]]

        ##### TODO: Preprocessing #####
        # Hint: How to prevent model from learning something it should not learn

        if self.split == "train":
            answer_start = tokenized_paragraph.char_to_token(question["answer_start"])
            answer_end = tokenized_paragraph.char_to_token(question["answer_end"])
            answer_center = (answer_start + answer_end) // 2

            # 随机偏移
            shift_range = self.max_paragraph_len // 3
            random_shift = random.randint(-shift_range, shift_range)

            # 初步窗口
            paragraph_start = answer_center - self.max_paragraph_len // 2 + random_shift
            paragraph_start = max(0, min(paragraph_start, len(tokenized_paragraph) - self.max_paragraph_len))
            paragraph_end = paragraph_start + self.max_paragraph_len

            # ✅ 校正步骤：确保答案完全在窗口内（加一点缓冲区）
            buffer = 10
            if answer_start < paragraph_start:
                paragraph_start = max(0, answer_start - buffer)
                paragraph_end = paragraph_start + self.max_paragraph_len
            if answer_end > paragraph_end:
                paragraph_end = min(len(tokenized_paragraph), answer_end + buffer)
                paragraph_start = max(0, paragraph_end - self.max_paragraph_len)
            if paragraph_end > len(tokenized_paragraph):
                paragraph_end = len(tokenized_paragraph)
                paragraph_start = max(0, paragraph_end - self.max_paragraph_len)

            # ===== 构造输入序列 =====
            input_ids_question = [101] + tokenized_question.ids[:self.max_question_len] + [102]
            input_ids_paragraph = tokenized_paragraph.ids[paragraph_start:paragraph_end] + [102]

            # ===== 答案 token 位置转换 =====
            # 原始 token index 转换到当前窗口（注意 +len(question)+2 是因为 CLS, SEP）
            answer_start_token = answer_start - paragraph_start + len(input_ids_question)
            answer_end_token = answer_end - paragraph_start + len(input_ids_question)

            # ===== Padding =====
            input_ids, token_type_ids, attention_mask = self.padding(input_ids_question, input_ids_paragraph)

            # ===== Return =====
            return (
                torch.tensor(input_ids),
                torch.tensor(token_type_ids),
                torch.tensor(attention_mask),
                torch.tensor(answer_start_token),
                torch.tensor(answer_end_token)
            )

        # Validation/Testing
        else:
            input_ids_list, token_type_ids_list, attention_mask_list = [], [], []
            
            # Paragraph is split into several windows, each with start positions separated by step "doc_stride"
            for i in range(0, len(tokenized_paragraph), self.doc_stride):
                
                # Slice question/paragraph and add special tokens (101: CLS, 102: SEP)
                input_ids_question = [101] + tokenized_question.ids[:self.max_question_len] + [102]
                input_ids_paragraph = tokenized_paragraph.ids[i : i + self.max_paragraph_len] + [102]
                
                # Pad sequence and obtain inputs to model
                input_ids, token_type_ids, attention_mask = self.padding(input_ids_question, input_ids_paragraph)
                
                input_ids_list.append(input_ids)
                token_type_ids_list.append(token_type_ids)
                attention_mask_list.append(attention_mask)
            
            return torch.tensor(input_ids_list), torch.tensor(token_type_ids_list), torch.tensor(attention_mask_list)


    def padding(self, input_ids_question, input_ids_paragraph):
        # Pad zeros if sequence length is shorter than max_seq_len
        padding_len = self.max_seq_len - len(input_ids_question) - len(input_ids_paragraph)
        # Indices of input sequence tokens in the vocabulary
        input_ids = input_ids_question + input_ids_paragraph + [0] * padding_len
        # Segment token indices to indicate first and second portions of the inputs. Indices are selected in [0, 1]
        token_type_ids = [0] * len(input_ids_question) + [1] * len(input_ids_paragraph) + [0] * padding_len
        # Mask to avoid performing attention on padding token indices. Mask values selected in [0, 1]
        attention_mask = [1] * (len(input_ids_question) + len(input_ids_paragraph)) + [0] * padding_len
        
        return input_ids, token_type_ids, attention_mask
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))   
from dataset.predata import dev_questions,dev_paragraphs_tokenized,dev_questions_tokenized,cfg
from dataset.predata import train_questions,train_paragraphs_tokenized,train_questions_tokenized
from dataset.predata import test_questions,test_paragraphs_tokenized,test_questions_tokenized

train_set = QA_Dataset("train", train_questions, train_questions_tokenized, train_paragraphs_tokenized)
dev_set = QA_Dataset("dev", dev_questions, dev_questions_tokenized, dev_paragraphs_tokenized)
test_set = QA_Dataset("test", test_questions, test_questions_tokenized, test_paragraphs_tokenized)

train_batch_size = cfg.train_batch_size

# Note: Do NOT change batch size of dev_loader / test_loader !
# Although batch size=1, it is actually a batch consisting of several windows from the same QA pair
train_loader = DataLoader(train_set, batch_size=train_batch_size, shuffle=True, pin_memory=True)
dev_loader = DataLoader(dev_set, batch_size=1, shuffle=False, pin_memory=True)
test_loader = DataLoader(test_set, batch_size=1, shuffle=False, pin_memory=True)
     
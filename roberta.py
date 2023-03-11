import pandas as pd
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification,Trainer, TrainingArguments,\
    AutoTokenizer, DebertaForSequenceClassification


import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm
import os
os.environ["WANDB_DISABLED"] = "true"
os.environ["TOKENIZERS_PARALLELISM"]="false"


# Read The data
training_set = pd.read_json('./data/train_set.json')
test_set = pd.read_json('./data/test_set.json')

max_length = 256

# load model and tokenizer and define length of the text sequence
model = RobertaForSequenceClassification.from_pretrained('roberta-base')
tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base', max_length = max_length)


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


# define the training arguments
training_args = TrainingArguments(
    output_dir = 'results_roberta',
    num_train_epochs=100,
    per_device_train_batch_size = 16, # 32
    gradient_accumulation_steps = 4,
    per_device_eval_batch_size= 16,
    evaluation_strategy = "steps",
    disable_tqdm = False,
    load_best_model_at_end=True,
    warmup_steps=1000,
    weight_decay=0.01,
    logging_steps = 1000,
    fp16 = True,
    logging_dir='logs',
    save_steps=1000,
    dataloader_num_workers = 8,
)

class NewsGroupsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor([self.labels[idx]])
        return item

    def __len__(self):
        return len(self.labels)


if __name__ == '__main__':

    train_test_split = 100 #

    train_encodings = tokenizer(training_set['text'].to_list()[train_test_split:], truncation=True, padding=True,
                                    max_length=max_length)
    test_encodings = tokenizer(training_set['text'].to_list()[0:train_test_split], truncation=True, padding=True,
                               max_length=max_length)

    train_y = training_set['label'].to_list()[train_test_split:]
    test_y = training_set['label'].to_list()[0:train_test_split]

    # convert our tokenized data into a torch Dataset
    train_dataset = NewsGroupsDataset(train_encodings, train_y)
    test_dataset = NewsGroupsDataset(test_encodings, test_y)

    # instantiate the trainer class and check for available devices
    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=test_dataset
    )
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # train the model
    trainer.train()

    # evaluate the current model after training
    trainer.evaluate()

    # saving the finetuned model & tokenizer
    model_path = "./model_roberta"
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
import pandas as pd
import csv
import random
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, accuracy_score
from transformers import (set_seed,
                          TrainingArguments,
                          Trainer,
                          GPT2Config,
                          GPT2Tokenizer,
                          AdamW,
                          get_linear_schedule_with_warmup,
                          GPT2ForSequenceClassification)

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")  # GPU acc on mac : "mps"

# Read The data
training_set = pd.read_json('./data/train_set.json')
test_set = pd.read_json('./data/test_set.json')

model_name_or_path= "gpt2"
n_labels = 2
max_length = 128
train_test_split = 3500


def load_model():

    # Get model configuration.
    print('Loading configuraiton...')
    model_config = GPT2Config.from_pretrained(pretrained_model_name_or_path=model_name_or_path, num_labels=n_labels)

    # Get model's tokenizer.
    print('Loading tokenizer...')
    tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model_name_or_path=model_name_or_path)
    # default to left padding
    tokenizer.padding_side = "left"
    # Define PAD Token = EOS Token = 50256
    tokenizer.pad_token = tokenizer.eos_token

    # Get the actual model.
    print('Loading model...')
    model = GPT2ForSequenceClassification.from_pretrained(pretrained_model_name_or_path=model_name_or_path, config=model_config)

    # resize model embedding to match new tokenizer
    model.resize_token_embeddings(len(tokenizer))

    # fix model padding token id
    model.config.pad_token_id = model.config.eos_token_id
    model.to(device)
    return tokenizer, model


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


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    # calculate accuracy using sklearn's function
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
    }


def get_prediction(tokenizer, text):
    # prepare our text into tokenized sequence
    inputs = tokenizer(text, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to("cuda")
    # perform inference to our model
    outputs = model(**inputs)
    # get output probabilities by doing softmax
    probs = outputs[0].softmax(1)
    # executing argmax function to get the candidate label
    return probs.argmax()


if __name__ == '__main__':

    tokenizer, model = load_model()
    train_encodings = tokenizer(training_set['text'].to_list()[0:train_test_split], truncation=True, padding=True,
                                max_length=max_length)
    test_encodings = tokenizer(training_set['text'].to_list()[train_test_split:], truncation=True, padding=True,
                               max_length=max_length)

    train_y = training_set['label'].to_list()[0:train_test_split]
    test_y = training_set['label'].to_list()[train_test_split:]

    # convert our tokenized data into a torch Dataset
    train_dataset = NewsGroupsDataset(train_encodings, train_y)
    test_dataset = NewsGroupsDataset(test_encodings, test_y)

    training_args = TrainingArguments(
        output_dir='./results_gpt',  # output directory
        num_train_epochs=50,  # total number of training epochs
        per_device_train_batch_size=24,  # batch size per device during training
        per_device_eval_batch_size=16,  # batch size for evaluation
        warmup_steps=500,  # number of warmup steps for learning rate scheduler
        weight_decay=0.001,  # strength of weight decay
        logging_dir='./logs',  # directory for storing logs
        load_best_model_at_end=True,  # load the best model when finished training (default metric is loss)
        # but you can specify `metric_for_best_model` argument to change to accuracy or other metric
        logging_steps=500,  # log & save weights each logging_steps
        save_steps=2000,
        evaluation_strategy="steps",  # evaluate each `logging_steps`
    )

    trainer = Trainer(
        model=model,  # the instantiated Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=train_dataset,  # training dataset
        eval_dataset=test_dataset,  # evaluation dataset
        compute_metrics=compute_metrics,  # the callback that computes metrics of interest
    )

    print("start training...")
    trainer.train()

    # evaluate the current model after training
    trainer.evaluate()

    # saving the finetuned model & tokenizer
    model_path = "./model_gpt"
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)

    text = """
    A black hole is a place in space where gravity pulls so much that even light can not get out. 
    The gravity is so strong because matter has been squeezed into a tiny space. This can happen when a star is dying.
    Because no light can get out, people can't see black holes. 
    They are invisible. Space telescopes with special tools can help find black holes. 
    The special tools can see how stars that are very close to black holes act differently than other stars.
    """

    print(get_prediction(text))

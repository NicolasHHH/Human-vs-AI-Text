import pandas as pd
import csv
from tqdm import tqdm
import os
os.environ["WANDB_DISABLED"] = "true"
os.environ["TOKENIZERS_PARALLELISM"]="false"

test_set = pd.read_json('./data/test_set.json')

from transformers import BertTokenizer, BertForSequenceClassification, \
    TrainingArguments, Trainer, XLNetForSequenceClassification, \
    AutoTokenizer, GPT2Config, GPT2Tokenizer, GPT2ForSequenceClassification, \
    RobertaTokenizerFast, RobertaForSequenceClassification, Trainer, TrainingArguments, DebertaForSequenceClassification

# model_name = "bert-base-uncased" #   bert-base-uncased" # "bert-base-cased" "xlnet-base-cased"
# tokenizer = AutoTokenizer.from_pretrained(model_name)

max_length = 256
#tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base', max_length = max_length)
tokenizer = AutoTokenizer.from_pretrained("./model_deberta")
predictions = []

#model = BertForSequenceClassification.from_pretrained("./model/").to("cuda")
#model = RobertaForSequenceClassification.from_pretrained('./model_roberta').to("cuda")
model = DebertaForSequenceClassification.from_pretrained("./model_deberta", num_labels=2).to("cuda")
# model = XLNetForSequenceClassification.from_pretrained("./model_xlnet/").to("cuda")

# model_name_or_path = "./model/"
# model_config = GPT2Config.from_pretrained(pretrained_model_name_or_path=model_name_or_path, num_labels=2)
# tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model_name_or_path=model_name_or_path)
# tokenizer.padding_side = "left"
# tokenizer.pad_token = tokenizer.eos_token
# model = GPT2ForSequenceClassification.from_pretrained(pretrained_model_name_or_path=model_name_or_path, config=model_config)
# model.resize_token_embeddings(len(tokenizer))
# model.config.pad_token_id = model.config.eos_token_id
# model.to("cuda")


def get_prediction(text):
    # prepare our text into tokenized sequence
    inputs = tokenizer(text, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to("cuda")
    # perform inference to our model
    outputs = model(**inputs)
    # get output probabilities by doing softmax
    probs = outputs[0].softmax(1)
    # executing argmax function to get the candidate label
    return probs.argmax()


for text in tqdm(test_set['text'].to_list()):
    predictions.append(get_prediction(text).item())

# Write predictions to a file
with open("submission_deberta.csv", "w") as pred:
    csv_out = csv.writer(pred)
    csv_out.writerow(['id', 'label'])
    for i, row in enumerate(predictions):
        csv_out.writerow([i, row])
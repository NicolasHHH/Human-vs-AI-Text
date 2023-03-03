import pandas as pd
import csv
from tqdm import tqdm

test_set = pd.read_json('./data/test_set.json')

from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments, Trainer

model_name = "bert-base-cased"
tokenizer = BertTokenizer.from_pretrained(model_name)
max_length = 128

predictions = []

model = BertForSequenceClassification.from_pretrained("./model/").to("cuda")


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
with open("submission.csv", "w") as pred:
    csv_out = csv.writer(pred)
    csv_out.writerow(['id', 'label'])
    for i, row in enumerate(predictions):
        csv_out.writerow([i, row])
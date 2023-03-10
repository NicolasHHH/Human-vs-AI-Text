# Human-vs-AI-Text

Kaggle competition : Build a model to predict whether a text is produced by a human or a machine


## Dataset description

`train_set.json`: This file contains 4000 paragraphs for various subjects (in the field text of the json file) and labels (in the field label of the json file). The dataset is divided as follows: 2016 human written text and 1984 text generated from different text generation models.

`test_set.json`: This file contains 4000 text in total, divided as follows: 2020 human written text and 1980 text generated using the same models used in the train_set. This dataset is distributed equally between the public and private leaderboards on kaggle.

## Quickstart 

### Train 
    python basic_transformer.py # bert_base_uncased

### Test 
    python inference.py # write results to submission.csv

## Submit Logs:

3/3
- tfidf(max_feature=10000) + svc(default) : **0.6175**  
- tfidf(max_feature=10000) + adaboost(default) : **0.598**  
- "bert-base-cased"(max_length=128) + BertForSequenceClassification(ep=30 spl=1600): **0.800** 
- "bert-base-cased"(max_length=128) + BertForSequenceClassification(ep=50 spl=2800): **0.830**

3/5
- "bert-base-cased"(max_length=128) + xlnet(ep=50 spl=3800): **0.79** 

### Weights

https://drive.google.com/drive/folders/1_38fv85i-WXuSAbz0ZwMnY5lVoMO_H8Y?usp=sharing
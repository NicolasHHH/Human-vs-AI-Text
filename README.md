# Human-vs-AI-Text

**Kaggle competition** : Build a model to predict whether a text is produced by a human or a machine

**Team Name** : HSZ

**Team Members** : 
- tianyang.huang@polytechnique.edu 
- yuyan.zhao@polytechnique.edu 
- biao.shi@polytechnique.edu

## Dataset description

`train_set.json`: This file contains 4000 paragraphs for various subjects (in the field text of the json file) and labels (in the field label of the json file). The dataset is divided as follows: 2016 human written text and 1984 text generated from different text generation models.

`test_set.json`: This file contains 4000 text in total, divided as follows: 2020 human written text and 1980 text generated using the same models used in the train_set. This dataset is distributed equally between the public and private leaderboards on kaggle.

## Quickstart 


### Best Model:
    DeBERTa+lgbm.ipynb

## Custom Training and Testing the Text Feature Extractor
### Train 
    python roberta.py # support roberta/deberta training
    python basic_transformer.py # support bert/xlnet training
    python gpt2.py # support gpt2 training

### Test 
    python inference.py # template for writing results to submission.csv

## Other Auxiliary Files
    xlnet.py # train xlnet (poor performance)
    EDA1.ipynb / EDA2.ipynb # Exploratory Data Analysis
    logistiv_regression_baseline.py # baseline with logistic regression / adaboost / random forest
    large_transformer.py # bert large (poor performance)
    semi-supervised # enhance the dataset base on leaderboard results. (not useful)


## Weights

Some of the best weights can be downloaded here : 

https://drive.google.com/drive/folders/1_38fv85i-WXuSAbz0ZwMnY5lVoMO_H8Y?usp=sharing

Once Downloaded, each model folder should be placed in parallel at the root of the project folder.

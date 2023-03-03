import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import csv
from sklearn.metrics import classification_report


# Read The data
training_set = pd.read_json('./data/train_set.json')
test_set = pd.read_json('./data/test_set.json')

# Use logistic regression to predict the class

# 1. word2vec
vectorizer = TfidfVectorizer(max_features=10000)
X = vectorizer.fit_transform(training_set['text'])

# 2. classification
clf = LogisticRegression()
clf = SVC()
clf = AdaBoostClassifier()
clf.fit(X, training_set['label'])
X_test = vectorizer.transform(test_set['text'])
predictions = clf.predict(X_test)


# Write predictions to a file
with open("submission.csv", "w") as pred:
    csv_out = csv.writer(pred)
    csv_out.writerow(['id','label'])
    for i, row in enumerate(predictions):
        csv_out.writerow([i, row])
        
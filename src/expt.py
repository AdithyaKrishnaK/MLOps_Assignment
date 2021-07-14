import pandas as pd
from pandas.io.parsers import read_csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn import metrics
import dvc.api
import pickle
import json

with dvc.api.open(repo="https://github.com/AdithyaKrishnaK/MLOps_Assignment", path="data/creditcard.csv", mode="r") as fd:
		df = pd.read_csv(fd)


train, test = train_test_split (df, test_size=0.2, random_state=42,stratify=df["Class"])

train.to_csv("../data/processed/train.csv")
test.to_csv("../data/processed/test.csv")

trainf = read_csv("../data/processed/train.csv")

X_train = trainf.drop(columns = ["Class"])
y_train = trainf["Class"]

model = tree.DecisionTreeClassifier(criterion="entropy")
model.fit(X_train,y_train)

pickle.dump(model,open("../models/model.pkl",'wb'))

testf = read_csv("../data/processed/test.csv")

X_test = testf.drop(columns = ["Class"])
y_test = testf["Class"]

y_predict = model.predict(X_test)

score = {
    "Accuracy":metrics.accuracy_score(y_test,y_predict),
    "F1 Score":metrics.f1_score(y_test,y_predict,average='weighted')
}

json_object = json.dumps(score, indent = 2)
  
with open("../metrics/acc_f1.json", "w") as outfile:
    outfile.write(json_object)
import pandas as pd
from pandas.io.parsers import read_csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn import metrics
import dvc.api
import pickle
import json

testf = read_csv("../data/processed/test.csv")
model = pickle.load("../models/model.pkl")

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
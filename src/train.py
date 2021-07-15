import pandas as pd
from pandas.io.parsers import read_csv
from sklearn.ensemble import RandomForestClassifier
import pickle

trainf = read_csv("../data/processed/train.csv")

X_train = trainf.drop(columns = ["Class"])
y_train = trainf["Class"]

model = RandomForestClassifier()
model.fit(X_train,y_train)

pickle.dump(model,open("../models/model.pkl",'wb'))
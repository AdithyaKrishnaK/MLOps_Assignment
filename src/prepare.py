import pandas as pd
from pandas.io.parsers import read_csv
from sklearn.model_selection import train_test_split
import dvc.api

with dvc.api.open(repo="https://github.com/AdithyaKrishnaK/MLOps_Assignment", path="data/creditcard.csv", mode="r") as fd:
		df = pd.read_csv(fd)


train, test = train_test_split (df, test_size=0.2, random_state=42,stratify=df["Class"])

train.to_csv("../data/processed/train.csv")
test.to_csv("../data/processed/test.csv")
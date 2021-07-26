import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("../data/creditcard.csv")

train, test = train_test_split (df, test_size=0.1, random_state=42,stratify=df["Class"])

test.to_csv("../data/test.csv")
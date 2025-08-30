import pandas as pd
import os

# columns: target,id,date,query,user,text
df = pd.read_csv("../data/raw/tweets.csv", encoding="latin-1", header=None)
df.columns = ["label","id","date","query","user","text"]

# keep only text + label
df = df[["text","label"]]

# Map labels 0,2,4 to 0,1,2
mapping = {0:0, 2:1, 4:2}  # 0=negative, 1=neutral, 2=positive
df["label"] = df["label"].map(mapping)

# make sure save directory exists
os.makedirs("../data/raw", exist_ok=True)

# save cleaned dataset
df.to_csv("../data/raw/tweets_clean.csv", index=False)
print(" Saved cleaned file to ../data/raw/tweets_clean.csv")

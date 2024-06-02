import os
import sys

import numpy as np
import pandas as pd
from pathlib import Path
# import torchaudio
import librosa
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def create_csv(data_path, save_path):
    data = []
    for path in tqdm(Path(data_path).glob("**/*.wav")):
        print(path)
        name = str(path).split('/')[-1].split('.')[0]
        label = str(path).split('/')[-2]
        
        try:
            # There are some broken files
            # s = torchaudio.load(path)
            s = librosa.load(path)
            # del s
            data.append({
                # "name": name,
                "path": path,
                "label": label
            })
        except Exception as e:
            print(str(path), e)
            pass

    df = pd.DataFrame(data)
    print(f"Step 0: {len(df)}")

    df["status"] = df["path"].apply(lambda path: True if os.path.exists(path) else None)
    df = df.dropna(subset=["path"])
    df = df.drop(columns=["status"])
    print(f"Step 1: {len(df)}")

    df = df.sample(frac=1)
    df = df.reset_index(drop=True)
    df.head()
    print("Labels: ", df["label"].unique())
    print()
    df.groupby("label").count()[["path"]]


    train_df, test_df = train_test_split(df, test_size=0.2, random_state=101, stratify=df["label"])

    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    train_df.to_csv(f"{save_path}/train.csv", sep="\t", encoding="utf-8", index=False)
    test_df.to_csv(f"{save_path}/test.csv", sep="\t", encoding="utf-8", index=False)


    print(train_df.shape)
    print(test_df.shape)
    
if __name__=="__main__":
    data_path = "data/sounds/"
    save_path = "data"
    create_csv(data_path, save_path)
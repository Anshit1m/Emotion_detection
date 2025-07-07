import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

def get_data():
    dataset = load_dataset("dair-ai/emotion")
    print("Type of dataset['train']:", type(dataset["train"]))
    df = dataset["train"].to_pandas()
    return train_test_split(df['text'], df['label'], test_size=0.2, random_state=42) 
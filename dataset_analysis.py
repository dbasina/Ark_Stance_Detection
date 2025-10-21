import os
import pandas as pd

dataset_path = "Dataset/weakly_labeled_race_dataset.csv"
dataset = pd.read_csv(dataset_path)

sample_data = dataset.iloc[1]
dataset.info()
 
print("Sample data:")
print(sample_data['text'])
print(sample_data['masked_text'])
print(sample_data['word'])

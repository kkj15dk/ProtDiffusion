import pandas as pd
from datasets import load_dataset

# Step 1: Load the CSV file as a dataset
dataset = load_dataset('csv', data_files='data.csv')
dataset.push_to_hub("kkj15dk/UniRef50")
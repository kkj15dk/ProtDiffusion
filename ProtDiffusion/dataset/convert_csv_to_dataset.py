# %%
from datasets import load_dataset

# Step 1: Load the CSV file as a dataset
dataset = load_dataset('csv', data_files='/home/kaspe/ProtDiffusion/datasets/SPARQL_UniRefALL.csv')
print(dataset)
print(dataset['train'][0])
print(dataset.column_names)
print(dataset.features)

# %%

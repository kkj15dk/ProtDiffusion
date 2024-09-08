# %%
import re
from datasets import load_dataset, Dataset
from collections import defaultdict
from tqdm import tqdm

# Define the transformation function for batches
def transform_length_batch(batch):
    # Use regular expression to extract the numeric part for each example in the batch
    lengths = []
    for length in batch['length']:
        match = re.match(r"(\d+)\^\^<http://www.w3.org/2001/XMLSchema#int>", length)
        if match:
            lengths.append(int(match.group(1)))
        else:
            raise ValueError(f"Invalid length: {length}")
    batch['length'] = lengths
    return batch

# Load the dataset
dataset = load_dataset('csv', data_files='/home/kkj/ProtDiffusion/datasets/SPARQL_UniRef50.csv')

# Rename columns as needed
dataset = dataset.rename_column(' sequence', 'sequence')
dataset = dataset.rename_column(' familytaxonid', 'familytaxonid')
dataset = dataset.rename_column(' proteinid', 'proteinid')
dataset = dataset.rename_column(' length', 'length')

# Apply the transformation to the 'length' column in batches
dataset = dataset.map(transform_length_batch, batched=True)

# Print the column names and a sample to verify the transformation
print(dataset.column_names)
print(dataset['train'][0])

# %%
def group_data(dataset: Dataset) -> Dataset:
    print("Grouping data")
    dataset_dict = defaultdict(lambda: {'familytaxonid': None, 'sequences': [], 'lengths': []})
    tqdm_dataset = tqdm(dataset, desc="Grouping data")
    
    for example in tqdm_dataset:
        id = example['clusterid']
        if dataset_dict[id]['familytaxonid'] is None:
            dataset_dict[id]['familytaxonid'] = example['familytaxonid']
        dataset_dict[id]['sequences'].append(example['sequence'])
        dataset_dict[id]['lengths'].append(example['length'])
    
    print(f"Grouped data into {len(dataset_dict)} clusters based on the identifier")

    # Create a generator function
    def data_generator():
        for id, data in tqdm(dataset_dict.items(), desc="Creating grouped dataset"):
            yield {
                'id': id,
                'label': data['familytaxonid'],
                'sequences': data['sequences'],
                'lengths': data['lengths']
            }

    grouped_dataset = Dataset.from_generator(data_generator)
    return grouped_dataset
# %%
dataset = group_data(dataset['train'])
# %%

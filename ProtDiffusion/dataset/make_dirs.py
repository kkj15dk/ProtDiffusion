import os
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
import gc

def make_directories(output_dir):
    dataset = load_dataset('/home/kkj/ProtDiffusion/datasets/SPARQL_UniRefALL', split='train', streaming=True).select_columns(['cluster50id','kingdom'])
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    for example in tqdm(dataset, total = 200915939, desc='Making directories'):
        id = example['cluster50id']
        kingdom = example['kingdom']
        file_path = os.path.join(output_dir, f'{id}.csv')
        new_row = {
            'cluster50id': id,
            'kingdom': kingdom,
            'proteinid': [],
            'sequence': [],
            'length': []
        }
        pd.DataFrame([new_row]).to_csv(file_path, index=False)
        
        # Explicitly delete variables and trigger garbage collection
        del example
        del id
        del kingdom
        del file_path
        del new_row

# Example usage:
output_dir = '/home/kkj/ProtDiffusion/datasets/SPARQL_UniRefALL_grouped50'
make_directories(output_dir)
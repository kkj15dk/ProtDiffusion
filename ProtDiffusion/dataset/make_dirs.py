import os
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
import gc

def make_directories(output_dir, chunk_size = 10000):
    chunks = pd.read_csv('/home/kaspe/ProtDiffusion/datasets/UniRef50_cluster_ids_and_kingdom.csv', chunksize=chunk_size)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    for chunk in tqdm(chunks, total = 50000000//chunk_size, desc='Making directories'): # Total len is approximate
        for example in chunk.iterrows():
            cluster_id = example[1]['cluster50id']
            kingdom = example[1][' kingdomid']
            if kingdom == 2: # Bacteria
                kingdom = 0
            elif kingdom == 2759: # Eukaryota
                kingdom = 1
            else:
                raise ValueError(f'Unknown kingdom: {kingdom}')
            file_path = os.path.join(output_dir, f'{cluster_id}.csv')
            new_row = {
                'cluster50id': cluster_id,
                'kingdom': kingdom,
                'proteinid': [],
                'input_ids': [],
                'length': []
            }
            pd.DataFrame([new_row]).to_csv(file_path, index=False)
        del chunk
        gc.collect()

# Example usage:
output_dir = '/home/kaspe/ProtDiffusion/datasets/SPARQL_UniRefALL_grouped50_inital'
make_directories(output_dir)
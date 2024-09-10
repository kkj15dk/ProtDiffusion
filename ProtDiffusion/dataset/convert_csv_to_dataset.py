# # %%
# import re
# from datasets import load_dataset, Dataset
# from collections import defaultdict
# from tqdm import tqdm
# import json
# import os
# import pandas as pd
# import dask.dataframe as dd
# from dask.distributed import Client
# from dask.diagnostics import ProgressBar

# # %%

# # Load the dataset
# dataset = load_dataset('csv', data_files='/home/kaspe/ProtDiffusion/datasets/SPARQL_UniRefALL.csv')

# # %%
# # Define the transformation function for batches
# def transform_length_kingdom_batch(batch):
#     # Use regular expression to extract the numeric part for each example in the batch
#     lengths = []
#     for length in batch['length']:
#         match = re.match(r"(\d+)\^\^<http://www.w3.org/2001/XMLSchema#int>", length)
#         if match:
#             lengths.append(int(match.group(1)))
#         else:
#             raise ValueError(f"Invalid length: {length}")
#     batch['length'] = lengths
#     kingdoms = []
#     for kingdom in batch['kingdom']:
#         if kingdom == 2:
#             kingdoms.append(0) # Prokaryota
#         elif kingdom == 2759:
#             kingdoms.append(1) # Eukaryota
#         else:
#             raise ValueError(f"Invalid kingdom: {kingdom}")
#     batch['kingdom'] = kingdoms
#     return batch

# # %%
# # Rename columns as needed
# dataset = dataset.rename_column(' sequence', 'sequence')
# dataset = dataset.rename_column(' kingdomid', 'kingdom')
# dataset = dataset.rename_column(' proteinid', 'proteinid')
# dataset = dataset.rename_column(' length', 'length')
# dataset = dataset.rename_column(' cluster90id', 'cluster90id')
# dataset = dataset.rename_column(' cluster50id', 'cluster50id')

# # %%
# # Apply the transformation to the 'length' and 'kingdom' column in batches
# dataset = dataset.map(transform_length_kingdom_batch, batched=True)

# # %%
# dataset.save_to_disk('/home/kaspe/ProtDiffusion/datasets/SPARQL_UniRefALL')

# # %%
# import pandas as pd
# from datasets import load_dataset
# from tqdm import tqdm
# import os
# import concurrent.futures
# import multiprocessing
# from filelock import FileLock
# import gc

# # %%
# def make_dirs(ids: list,
#               output_dir='/home/kkj/ProtDiffusion/datasets/SPARQL_UniRefALL_grouped50'
# ):

#     for id50 in tqdm(ids, desc='Creating directories'):
#         file_path = os.path.join(output_dir, f'{id50}.csv')
#         lock_path = file_path + '.lock'
#         with FileLock(lock_path):
#             new_row = {
#                 'cluster50id': id50,
#                 'kingdom': None,
#                 'proteinid': [],
#                 'sequence': [],
#                 'length': []
#             }
#             pd.DataFrame([new_row]).to_csv(file_path, index=False)
        
#         del id50
#         del file_path
#         del lock_path

# def process_example(example, 
#                     output_dir, 
#                     id_key, 
#                     sequence_key, 
#                     length_key, 
#                     proteinid_key
# ): 
    
#     example_id = example[id_key] 
#     file_path = os.path.join(output_dir, f'{example_id}.csv')
#     lock_path = file_path + '.lock'

#     with FileLock(lock_path):
#         current_df = pd.read_csv(file_path)
#         current_df[proteinid_key] = current_df[proteinid_key].apply(eval)
#         current_df[sequence_key] = current_df[sequence_key].apply(eval)
#         current_df[length_key] = current_df[length_key].apply(eval)
        
#         if example[sequence_key] not in current_df.at[0, sequence_key]:
#             current_df.at[0, sequence_key].append(example[sequence_key])
#             current_df.at[0, length_key].append(example[length_key])
#             current_df.at[0, proteinid_key].append(example[proteinid_key])
            
#             current_df.to_csv(file_path, index=False)
#     del example
#     del current_df

# def process_chunk(indexes, dataset, output_dir, id_key, sequence_key, length_key, proteinid_key):
#     for idx in indexes:
#         example = dataset[idx]
#         process_example(example, output_dir, id_key, sequence_key, length_key, proteinid_key)
#         del example

# def group_dataset(dataset, chunk_size, output_dir='/home/kaspe/ProtDiffusion/datasets/SPARQL_UniRefALL_grouped50', id_key='cluster50id', sequence_key='sequence', length_key='length', proteinid_key='proteinid'): 
#     if not os.path.exists(output_dir): 
#         os.makedirs(output_dir)

#     with tqdm(total=len(dataset)) as pbar:
#         with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
#             futures = (executor.submit(process_chunk, list(range(i, min(i + chunk_size, len(dataset)))), dataset, output_dir, id_key, sequence_key, length_key, proteinid_key) for i in range(0, len(dataset), chunk_size))
#             for future in concurrent.futures.as_completed(futures):
#                 future.result()
#                 pbar.update(chunk_size)

# %%
import os
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
import gc

def make_directories(output_dir):
    dataset = load_dataset('/home/kkj/ProtDiffusion/datasets/SPARQL_UniRefALL', split='train').select_columns(['cluster50id','kingdom'])
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    for example in tqdm(dataset, total = 200915939, desc='Loading ids'):
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
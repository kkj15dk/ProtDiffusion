# %%
from datasets import load_dataset, load_from_disk, Dataset
from tqdm import tqdm
import os
import pandas as pd
import numpy as np
from transformers import PreTrainedTokenizerFast
from ProtDiffusion.training_utils import round_length

# %%
# Define the parameters
sequence_key = 'sequence'
id_key = 'clusterid' # This is the column to group by
label_key = 'familytaxonid'
pad_to_multiple_of = 16
output_path = '/home/kkj/ProtDiffusion/datasets/'
# input_path = '/home/kkj/ProtDiffusion/datasets/UniRef50_sorted.csv' # Has to be sorted by id
input_path = '/home/kkj/ProtDiffusion/datasets/testcase-UniRef50_sorted.csv'
filename_encoded = 'UniRef50-test-good?'
filename_grouped = 'UniRef50-test-good?'

# %%
# Define the transformation function for batches
def preprocess(example: dict,
               sequence_key: str = 'sequence', 
               label_key: str = 'familytaxonid', 
               pad_to_multiple_of=pad_to_multiple_of,
):
    sequence = example[sequence_key]
    label = example[label_key]
    if label == 2:
        label = 0
    elif label == 2759:
        label = 1
    else:
        raise ValueError(f"Invalid label: {label}")
    length = round_length(len(sequence), rounding=pad_to_multiple_of)
    return {'sequence': sequence, 'label': label, 'length': length}

def list_to_np(chunk):
    chunk['label'] = chunk['label'].apply(lambda x: np.array(x).astype(np.int16))
    chunk['length'] = chunk['length'].apply(lambda x: np.array(x).astype(np.int16))
    chunk['sequence'] = chunk['sequence'].apply(lambda x: np.array(x).astype(np.bytes_))
    return chunk

def stream_groupby_gen(dataset: Dataset, 
                       id_key: str, 
                       chunk_size=100000, 
):
    '''
    Input:
    A dataset with columns 'sequence', 'label', 'length', and id_key. id_key is the column to group by, and will be renamed to 'id'.
    '''
    agg = lambda chunk: chunk.groupby('id').agg({
        'label': list, 
        'length': list,
        'sequence': list
        })

    # Tell pandas to read the data in chunks
    chunks = dataset.rename_column(id_key, 'id').select_columns(['id','label','length','sequence']).to_pandas(batched=True, batch_size=chunk_size)
    
    orphans = pd.DataFrame()

    for chunk in tqdm(chunks, desc='Processing chunks', unit='chunk', total=len(dataset)//chunk_size):

        # Add the previous orphans to the chunk
        chunk = pd.concat((orphans, chunk))

        # Determine which rows are orphans
        last_val = chunk['id'].iloc[-1]
        is_orphan = chunk['id'] == last_val

        # Put the new orphans aside
        chunk, orphans = chunk[~is_orphan], chunk[is_orphan]
        # Perform the aggregation and store the results
        chunk = agg(chunk)
        chunk = list_to_np(chunk)

        for i in range(len(chunk)):
            data = chunk.iloc[i]
            yield {'id': data.name, 'label': data['label'], 'length': data['length'], 'sequence': data['sequence']}


    # Don't forget the remaining orphans
    if len(orphans):
        chunk = agg(orphans)
        chunk = list_to_np(chunk)

        for i in range(len(chunk)):
            data = chunk.iloc[i]
            yield {'id': data.name, 'label': data['label'], 'length': data['length'], 'sequence': data['sequence']}

# %%
# Load the dataset
dataset = load_dataset('csv', data_files=input_path)['train']

# %%
# dataset = dataset.rename_column(' kingdomid', 'familytaxonid')
# dataset = dataset.rename_column(' sequence', 'sequence')
# dataset = dataset.rename_column(' cluster90id', 'cluster90id')
# dataset = dataset.rename_column(' cluster100id', 'cluster100id')

# %%
# filter so that only sequences with ACDEFGHIKLMNOPQRSTUVWY are included
# dataset = dataset.filter(lambda x: all(c in 'ACDEFGHIKLMNPQRSTVWY' for c in x['sequence']), num_proc=12)

# %%
# Encode the dataset
if not os.path.exists(f'{output_path}{filename_encoded}'):
    print(f"Encoding {filename_encoded}")
    dataset = dataset.map(preprocess, 
                            fn_kwargs={'sequence_key': sequence_key, 
                                       'label_key': label_key,
                                       'pad_to_multiple_of': pad_to_multiple_of,
                            },
                            remove_columns=[label_key],
                            batched=False, 
                            num_proc=12,
    )
    dataset.save_to_disk(f'{output_path}{filename_encoded}')
else:
    print(f"{filename_encoded} already encoded")

# %%
# Group by the id column and aggregate the input_ids, labels, and lengths
if not os.path.exists(f'{output_path}{filename_grouped}_grouped'):
    print(f"Grouping {filename_grouped}")
    dataset = load_from_disk(f'{output_path}{filename_encoded}')
    print("Loaded dataset, starting grouping")
    dataset = Dataset.from_generator(stream_groupby_gen, 
                                     gen_kwargs={'dataset': dataset, 'id_key': id_key},
    ).with_format('numpy')
    print("Grouping done, saving to disk")
    dataset.save_to_disk(f'{output_path}{filename_grouped}_grouped')
else:
    print(f"{filename_grouped} already grouped")

print('Doen')

# %%
# Load the grouped dataset
dataset = load_from_disk(f'{output_path}{filename_grouped}_grouped')
print(dataset)
print(dataset[0])
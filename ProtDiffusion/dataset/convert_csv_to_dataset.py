# %%
import re
from datasets import load_dataset, Dataset
from tqdm import tqdm
import os
import pandas as pd
from transformers import PreTrainedTokenizerFast

# %%
# Define the transformation function for batches
def encode(example, 
            tokenizer: PreTrainedTokenizerFast, 
            sequence_key='sequence', 
            id_key='clusterid', 
            label_key='kingdom', 
            pad_to_multiple_of=16
):
    # Use regular expression to extract the numeric part for each example in the batch
    output = tokenizer(example[sequence_key],
                        padding=True,
                        truncation=False, # We truncate the sequences later, so we set this to False
                        pad_to_multiple_of = pad_to_multiple_of,
                        return_token_type_ids=False,
                        return_attention_mask=False, # We need to attend to padding tokens, so we set this to False
    )
    output['id'] = example[id_key]
    label = example[label_key]
    if label == 2:
        output['label'] = 0
    elif label == 2759:
        output['label'] = 1
    else:
        raise ValueError(f"Invalid label: {label}")
    output['length'] = len(output['input_ids'])
    return output

def agg_fn(chunk: pd.DataFrame, key: str) -> pd.DataFrame:
    chunk.groupby(key).agg({
        'label': 'first', 
        'length': list,
        'input_ids': list, 
        })

def stream_groupby_csv(dataset: Dataset, key: str, agg = agg_fn, chunk_size=1e6, output_path='output.csv'):

    # Tell pandas to read the data in chunks
    chunks = dataset.to_pandas(chunksize=chunk_size, batched=True)
    orphans = pd.DataFrame()

    for chunk in chunks:

        # Add the previous orphans to the chunk
        chunk = pd.concat((orphans, chunk))

        # Determine which rows are orphans
        last_val = chunk[key].iloc[-1]
        is_orphan = chunk[key] == last_val

        # Put the new orphans aside
        chunk, orphans = chunk[~is_orphan], chunk[is_orphan]

        # Perform the aggregation and store the results
        agg(chunk, key).to_csv(output_path, mode='a', header=False)

    # Don't forget the remaining orphans
    if len(orphans):
        agg(orphans, key).to_csv(output_path, mode='a', header=False)

# %%
# Load the dataset
dataset = load_dataset('csv', data_files='/home/kaspe/ProtDiffusion/datasets/SPARQL_UniRefALL.csv')

# %%
# Rename columns as needed
dataset = dataset.rename_column(' sequence', 'sequence')
dataset = dataset.rename_column(' kingdomid', 'kingdom')
dataset = dataset.rename_column('clusterid', 'clusterid')
tokenizer = PreTrainedTokenizerFast.from_pretrained("kkj15dk/protein_tokenizer_new")

# %%
# Apply the transformation to the 'length' and 'kingdom' column in batches
dataset = dataset.map(encode, 
                        fn_kwargs={'sequence_key': 'sequence', 
                                    'id_key': 'clusterid',
                                    'label_key': 'kingdom',
                                    'pad_to_multiple_of': 16, 
                                    'tokenizer': tokenizer},
                        batched=False)

# %%
dataset.save_to_disk('/home/kaspe/ProtDiffusion/datasets/SPARQL_UniRef50_encoded')

stream_groupby_csv(dataset, key='id', output_path='/home/kaspe/ProtDiffusion/datasets/SPARQL_UniRef50_encoded_grouped.csv')
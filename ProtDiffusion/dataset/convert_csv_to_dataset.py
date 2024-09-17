# %%
from datasets import load_dataset, load_from_disk, Dataset
from tqdm import tqdm
import os
import pandas as pd
from transformers import PreTrainedTokenizerFast

# %%
# Define the parameters
sequence_key = 'sequence'
id_key = 'cluster90id' # This is the column to group by
label_key = 'kingdomid'
pad_to_multiple_of = 16
output_path = '/home/kaspe/ProtDiffusion/datasets/'
input_path = '/home/kaspe/ProtDiffusion/datasets/UniRefALL_sorted.csv' # Has to be sorted by id
filename_encoded = 'UniRefALL'
filename_grouped = 'UniRef50'

# %%
# Define the transformation function for batches
def encode(example, 
            tokenizer: PreTrainedTokenizerFast, 
            sequence_key='sequence', 
            label_key='kingdom', 
            pad_to_multiple_of=pad_to_multiple_of,
):
    # Use regular expression to extract the numeric part for each example in the batch
    output = tokenizer(example[sequence_key],
                        padding=True,
                        truncation=False, # We truncate the sequences later, so we set this to False
                        pad_to_multiple_of = pad_to_multiple_of,
                        return_token_type_ids=False,
                        return_attention_mask=False, # We need to attend to padding tokens, so we set this to False
    )
    label = example[label_key]
    if label == 2:
        output['label'] = 0
    elif label == 2759:
        output['label'] = 1
    else:
        raise ValueError(f"Invalid label: {label}")
    output['length'] = len(output['input_ids'])
    return output

def stream_groupby_gen(dataset: Dataset, 
                       id_key: str, 
                       chunk_size=100000, 
):
    '''
    Input:
    A dataset with columns 'input_ids', 'label', 'length', and id_key. id_key is the column to group by, and will be renamed to 'id'.
    '''
    agg = lambda chunk: chunk.groupby('id').agg({
        'label': list, 
        'length': list,
        'input_ids': list
        })

    # Tell pandas to read the data in chunks
    chunks = dataset.rename_column(id_key, 'id').select_columns(['id','label','length','input_ids']).to_pandas(batched=True, batch_size=chunk_size)
    
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
        dataset = Dataset.from_pandas(chunk.reset_index())
        for i in range(len(dataset)):
            yield dataset[i]
    # Don't forget the remaining orphans
    if len(orphans):
        chunk = agg(orphans)
        dataset = Dataset.from_pandas(chunk.reset_index())
        for i in range(len(dataset)):
            yield dataset[i]

# # %%
# # Load the dataset
# dataset = load_dataset('csv', data_files=input_path)['train']
# tokenizer = PreTrainedTokenizerFast.from_pretrained("kkj15dk/protein_tokenizer_new")

# # %%
# dataset = dataset.rename_column(' kingdomid', 'kingdomid')
# dataset = dataset.rename_column(' sequence', 'sequence')
# dataset = dataset.rename_column(' cluster90id', 'cluster90id')
# dataset = dataset.rename_column(' cluster100id', 'cluster100id')

# %%
# Encode the dataset
if not os.path.exists(f'{output_path}{filename_encoded}_encoded'):
    print(f"Encoding {filename_encoded}")
    dataset = dataset.map(encode, 
                            fn_kwargs={'sequence_key': sequence_key, 
                                        'label_key': label_key,
                                        'pad_to_multiple_of': pad_to_multiple_of, 
                                        'tokenizer': tokenizer},
                            remove_columns=[sequence_key, label_key],
                            batched=False)
    dataset.save_to_disk(f'{output_path}{filename_encoded}_encoded')
else:
    print(f"{filename_encoded} already encoded")

# %%
# Group by the id column and aggregate the input_ids, labels, and lengths
if not os.path.exists(f'{output_path}{filename_grouped}_encoded_grouped'):
    print(f"Grouping {filename_grouped}")
    dataset = load_from_disk(f'{output_path}{filename_encoded}_encoded')
    print("Loaded dataset, starting grouping")
    dataset = Dataset.from_generator(stream_groupby_gen, gen_kwargs={'dataset': dataset, 'id_key': id_key})
    dataset.save_to_disk(f'{output_path}{filename_grouped}_encoded_grouped')
else:
    print(f"{filename_grouped} already grouped")

print('Doen')
# %%

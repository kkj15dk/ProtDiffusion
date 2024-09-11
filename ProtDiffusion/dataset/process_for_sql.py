# %%
from transformers import PreTrainedTokenizerFast
from datasets import load_dataset

def encode(example, pad_to_multiple_of: int, tokenizer: PreTrainedTokenizerFast):
    output = tokenizer(example['sequence'],
                        padding = True,
                        truncation=False, # We need to truncate the sequences later, so we set this to False
                        pad_to_multiple_of = pad_to_multiple_of,
                        return_token_type_ids=False,
                        return_attention_mask=False, # We need to attend to padding tokens, so we set this to False
    )
    output['cluster100id'] = example['cluster100id']
    output['cluster90id'] = example['cluster90id']
    output['cluster50id'] = example['cluster50id']
    output['sequence'] = example['sequence']
    output['kingdom'] = example['kingdom']
    output['length'] = len(output['input_ids'])
    return output
# %%
tokenizer = PreTrainedTokenizerFast.from_pretrained("kkj15dk/protein_tokenizer_new")

# %%
dataset = load_dataset('/home/kaspe/ProtDiffusion/datasets/SPARQL_UniRefALL', split='train')

# %%
# Encode the dataset
pad_to_multiple_of = 16
dataset = dataset.map(lambda example: encode(example, pad_to_multiple_of, tokenizer), batched=False)

# %%
dataset.save_to_disk('/home/kaspe/ProtDiffusion/datasets/SPARQL_UniRefALL_encoded')
# %%

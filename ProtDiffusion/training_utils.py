# %%
from dataclasses import dataclass
import os
from typing import Optional, Literal, Union, List, Tuple
import random
import pickle

import torch

from transformers import PreTrainedTokenizerFast
from datasets import load_from_disk, load_dataset

@dataclass
class TrainingConfig:
    batch_size: int = 2  # the batch size
    mega_batch: int = 1000 # how many batches to use for batchsampling
    num_epochs: int = 1  # the number of epochs to train the model
    gradient_accumulation_steps: int = 2  # the number of steps to accumulate gradients before taking an optimizer step
    learning_rate: float = 1e-4  # the learning rate
    lr_warmup_steps:int  = 1000
    save_image_model_steps:int  = 100
    mixed_precision: str = "fp16"  # `no` for float32, `fp16` for automatic mixed precision
    optimizer: str = "AdamW"  # the optimizer to use, choose between `AdamW`, `Adam`, `SGD`, and `Adamax`
    SGDmomentum: float = 0.9
    output_dir: str = os.path.join("output","protein-VAE-UniRef50-8")  # the model name locally and on the HF Hub
    pad_to_multiple_of: int = 16
    max_len: int = 32  # truncation of the input sequence

    class_embeddings_concat = False  # whether to concatenate the class embeddings to the time embeddings

    push_to_hub = False  # Not implemented yet. Whether to upload the saved model to the HF Hub
    hub_model_id = "kkj15dk/protein-VAE"  # the name of the repository to create on the HF Hub
    hub_private_repo = False
    overwrite_output_dir = True  # overwrite the old model when re-running the notebook
    seed: int = 42

    automatic_checkpoint_naming: bool = True  # whether to automatically name the checkpoints
    total_limit: int = 1  # the total limit of checkpoints to save

    cutoff: Optional[float] = None # cutoff for when to predict the token given the logits, and when to assign the unknown token 'X' to this position
    skip_special_tokens = False # whether to skip the special tokens when writing the evaluation sequences
    kl_weight: float = 0.05 # the weight of the KL divergence in the loss function

    weight_decay: float = 0.01 # weight decay for the optimizer
    grokfast: bool = False # whether to use the grokfast algorithm
    grokfast_alpha: float = 0.98 #Momentum hyperparmeter of the EMA.
    grokfast_lamb: float = 2.0 #Amplifying factor hyperparameter of the filter.


def prepare_dataset(dataset, 
                    dataset_split: str,
                    dataset_dir: str,
                    sequence_key: str = 'sequence',
):
    '''
    Prepare the dataset, save the lengths 
    '''
    dataset = dataset[dataset_split]
    lengths_path = os.path.join(dataset_dir, f'{dataset_split}_lengths.pkl')

    if os.path.exists(lengths_path):
        print(f'Loading lengths for {dataset_split}')
        with open(lengths_path, 'rb') as f:
            lengths = pickle.load(f)
    else:
        print(f'Calculating lengths for {dataset_split}')
        lengths = list(map(lambda x: len(x[sequence_key]), dataset))
        os.makedirs(dataset_dir, exist_ok=True)
        with open(lengths_path, 'wb') as f:
            pickle.dump(lengths, f)

    return dataset, lengths

def random_slice(encoded_batch, max_length): # This seems inefficient, but I don't know how to do it better, as the current tokenizer doesn't support RANDOM truncation
    batch_size, seq_length = encoded_batch['input_ids'].shape
    start_indices = torch.randint(0, seq_length - max_length + 1, (batch_size,))
    
    input_ids = torch.stack([encoded_batch['input_ids'][i, start:start + max_length] for i, start in enumerate(start_indices)])
    attention_mask = torch.stack([encoded_batch['attention_mask'][i, start:start + max_length] for i, start in enumerate(start_indices)])
    
    return input_ids, attention_mask


class BatchSampler:
    '''
    BatchSampler for variable length sequences, batching by similar lengths, to prevent excessive padding.
    '''
    def __init__(self, 
                 lengths, 
                 batch_size, 
                 mega_batch_size, 
                 tokenizer: PreTrainedTokenizerFast, 
                 max_lenght: Optional[int] = None,
                 pad_to_multiple_of: int = 16,
                 sequence_key: str = 'sequence', 
                 drop_last = True):
        self.lengths = lengths
        self.batch_size = batch_size
        self.mega_batch_size = mega_batch_size
        self.drop_last = drop_last
        self.tokenizer = tokenizer
        self.max_lenght = max_lenght
        self.pad_to_multiple_of = pad_to_multiple_of
        self.sequence_key = sequence_key

    def collate_fn(self, batch):
        sequences = [batch[i][self.sequence_key] for i in range(len(batch))]
        encoded_batch = self.tokenizer(sequences, 
                                       pad_to_multiple_of=self.pad_to_multiple_of,
                                       padding=True,
                                       truncation=False,
                                       return_attention_mask=True,
                                       return_token_type_ids=False,
                                       return_tensors='pt'
                                       )
        
        identifiers = [batch[i]['id'] for i in range(len(batch))]
        class_label = torch.tensor([batch[i]['class'] for i in range(len(batch))], dtype=torch.long)

        if self.max_lenght is None:
            input_ids = encoded_batch['input_ids']
            attention_mask = encoded_batch['attention_mask']
        else:
            input_ids, attention_mask = random_slice(encoded_batch, self.max_lenght)
        
        return {
            'id': identifiers,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'class_label': class_label,
        }
    
    def __iter__(self):
        size = len(self.lengths)
        indices = list(range(size))
        random.shuffle(indices)

        step = self.mega_batch_size * self.batch_size
        for i in range(0, size, step):
            pool = indices[i:i+step]
            pool = sorted(pool, key=lambda x: self.lengths[x])
            mega_batch_indices = list(range(0, len(pool), self.batch_size))
            random.shuffle(mega_batch_indices) # shuffle the mega batches, so that the model doesn't see the same order of lengths every time. The small batch will however always be the one with longest lengths
            for j in mega_batch_indices:
                if self.drop_last and j + self.batch_size > len(pool): # drop the last batch if it's too small
                    continue
                batch = pool[j:j+self.batch_size]
                random.shuffle(batch) # shuffle the batch, so that the model doesn't see the same order of lengths every time
                yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.lengths) // self.batch_size
        else:
            return (len(self.lengths) + self.batch_size - 1) // self.batch_size

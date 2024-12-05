# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
import torch.multiprocessing as mp
from ema_pytorch import EMA

from diffusers.optimization import get_cosine_schedule_with_warmup, get_constant_schedule_with_warmup
from diffusers.schedulers import DDPMScheduler
from datasets import Dataset

import os
import numpy as np
import gc
from copy import deepcopy
import math

from dataclasses import dataclass
from typing import Optional, Union, List

from transformers import PreTrainedTokenizerFast
import random

from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from tqdm import tqdm

import ot

from .models.autoencoder_kl_1d import AutoencoderKL1D, AutoencoderKLOutput1D
from .models.vae import EncoderKLOutput1D
from .models.dit_transformer_1d import DiTTransformer1DModel, Transformer1DModelOutput
from .models.pipeline_protein import ProtDiffusionPipeline, logits_to_token_ids
from .schedulers.FlowMatchingEulerScheduler import FlowMatchingEulerScheduler
from .visualization_utils import make_logoplot

# Set a random seed in a bunch of different places
def set_seed(seed: int = 42) -> None:
    """
    Set the random seed for reproducibility.

    Args:
        seed (int): The random seed to set.

    Returns:
        None
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"Random seed set as {seed}")

def get_cosine_10x_decay_schedule_with_warmup(
    optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int, num_cycles: float = 0.5, last_epoch: int = -1
) -> LambdaLR:
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 10 times lower lr, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        num_periods (`float`, *optional*, defaults to 0.5):
            The number of periods of the cosine function in a schedule (the default is to just decrease from the max
            value to 10 times lower lr following a half-cosine).
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return (9/20) * (11/9 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))

    return LambdaLR(optimizer, lr_lambda, last_epoch)

def get_cosine_100x_decay_schedule_with_warmup(
    optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int, num_cycles: float = 0.5, last_epoch: int = -1
) -> LambdaLR:
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 10 times lower lr, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        num_periods (`float`, *optional*, defaults to 0.5):
            The number of periods of the cosine function in a schedule (the default is to just decrease from the max
            value to 10 times lower lr following a half-cosine).
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return (99/200) * (101/99 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))

    return LambdaLR(optimizer, lr_lambda, last_epoch)

@dataclass
class VAETrainingConfig:
    batch_size: int = 64  # the batch size
    mega_batch: int = 1000 # how many batches to use for batchsampling
    num_epochs: int = 1  # the number of epochs to train the model
    gradient_accumulation_steps: int = 2  # the number of steps to accumulate gradients before taking an optimizer step
    learning_rate: float = 1e-4  # the learning rate
    lr_warmup_steps:int  = 1000
    lr_schedule: str = "cosine" # the learning rate schedule, choose between `constant`, `cosine`, `cosine_10x_decay`, and `cosine_100x_decay`
    save_image_model_steps:int  = 1000
    mixed_precision: str = "fp16"  # `no` for float32, `fp16` for automatic mixed precision
    optimizer: str = "AdamW"  # the optimizer to use, choose between `AdamW`, `Adam`, `SGD`, and `Adamax`
    SGDmomentum: float = 0.9
    output_dir: str = os.path.join("output","test")  # the model name locally and on the HF Hub
    pad_to_multiple_of: int = 16 # should be a multiple of 2 for each layer in the VAE.
    max_len: int = 512  # truncation of the input sequence
    max_len_start: Optional[int] = 64  # the starting length of the input sequence
    max_len_doubling_steps: Optional[int] = 100000  # the number of steps to double the input sequence length

    class_embeddings_concat = False  # whether to concatenate the class embeddings to the time embeddings

    push_to_hub = False  # Not implemented yet. Whether to upload the saved model to the HF Hub
    hub_model_id = "kkj15dk/protein-VAE"  # the name of the repository to create on the HF Hub
    hub_private_repo = False
    overwrite_output_dir = False  # overwrite the old model when re-running the notebook
    seed: int = 42

    automatic_checkpoint_naming: bool = True  # whether to automatically name the checkpoints
    total_checkpoints_limit: int = 5  # the total limit of checkpoints to save

    cutoff: Optional[float] = None # cutoff for when to predict the token given the logits, and when to assign the unknown token 'X' to this position
    skip_special_tokens = False # whether to skip the special tokens when writing the evaluation sequences
    kl_weight: float = 0.1 # the weight of the KL divergence in the loss function
    kl_warmup_steps: Optional[int] = None # the number of steps to warm up the KL divergence weight
    kl_schedule: str = 'constant_with_warmup' # choose between 'constant_with_warmup', where there is a warmup of kl_warmup_steps, or 'constant_with_restarts', where the warmup restarts after a constant region of kl_warmup_steps.

    gradient_clip_val: Optional[float] = 5.0  # the value to clip the gradients to
    weight_decay: float = 0.01 # weight decay for the optimizer
    ema_decay: float = 0.9999 # the decay rate for the EMA
    ema_update_after: int = 1000 # the number of steps to wait before updating the EMA
    ema_update_every: int = 1 # the number of steps to wait before updating the EMA

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

        if not self.overwrite_output_dir and os.path.exists(self.output_dir):
            raise ValueError("Output directory already exists. Set `config.overwrite_output_dir` to `True` to overwrite it.")
        
        if self.push_to_hub:
            raise NotImplementedError("Pushing to the HF Hub is not implemented yet")
        
        assert self.optimizer in ["AdamW", "Adam", "SGD", "Adamax"], "Invalid optimizer, choose between `AdamW`, `Adam`, `SGD`, and `Adamax`"
        assert self.mixed_precision in ["no", "fp16"], "Invalid mixed precision setting, choose between `no` and `fp16`" # TODO: implement fully
        assert self.max_len % self.pad_to_multiple_of == 0, "The maximum length of the input sequence must be a multiple of the pad_to_multiple_of parameter."
        assert self.max_len_start is None or self.max_len_start % self.pad_to_multiple_of == 0, "The starting length of the input sequence must be a multiple of the pad_to_multiple_of parameter."

        if self.max_len_start is not None:
            assert self.max_len_start <= self.max_len, "The starting length of the input sequence must be less than or equal to the maximum length of the input sequence, or None."


def count_parameters(model):
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {n_params} trainable parameters")
    return n_params

def round_length(length: int, pad: int = 2, rounding: int = 8) -> int:
    '''
    Round the length to what it will be after processing by the tokenization process.
    '''
    return int(np.ceil((length + pad) / rounding) * rounding)

def process_sequence(sequence: str,
                     bos_token: str = "[",
                     eos_token: str = "]",
                     pad_token: str = "-",
                     pad_to_multiple_of: int = 8,
                     generator: torch.Generator = None,
                     random_padding: bool = True,
    ) -> str:
        '''
        Process the sequence by adding the bos and eos tokens, and padding it to a multiple of 16 (or what the variable is set to in the round_kength).
        Return the sequence and the length of the sequence.
        '''
        seq_len = round_length(len(sequence), pad = 2, rounding = pad_to_multiple_of)
        sequence = bos_token + sequence + eos_token
        len_diff = seq_len - len(sequence)
        if len_diff == 0:
            return sequence
        if random_padding:
            rand_int = torch.randint(0, len_diff, (1,), generator = generator).item()
            sequence = pad_token * rand_int + sequence + pad_token * (len_diff - rand_int)
        else:
            sequence = sequence + pad_token * len_diff
        return sequence

def tokenize_sequence(seq_list: List[str],
                      tokenizer: PreTrainedTokenizerFast,
                      padding: bool = True,
                      truncation: bool = False,
                      return_token_type_ids: bool = False,
                      return_attention_mask: bool = False,
                      return_tensors: str = "pt",
    ) -> torch.Tensor:
    tokenized = tokenizer(seq_list,
                          padding=padding,
                          truncation=truncation,
                          return_token_type_ids=return_token_type_ids,
                          return_attention_mask=return_attention_mask,
                          return_tensors=return_tensors,
    )

    return tokenized
    

def calculate_stats(averages: List[int], standard_deviations: List[int], num_elements: List[int]):
    '''
    Calculate the mean and standard deviation of the latent space.
    '''
    summation = 0
    for average, n in zip(averages, num_elements):
        summation += average * n
    mu = summation / sum(num_elements)

    variance = 0
    for sd, n in zip(standard_deviations, num_elements):
        variance += sd ** 2 * (n-1)
    standard_dev = np.sqrt(variance / (sum(num_elements) - len(num_elements))) # TODO: Check if this is correct, maybe it should be -1 instead of -len(residues_in_group)

    return mu, standard_dev

def reorder_noise_for_OT(latent: torch.Tensor, noise: torch.Tensor, debug: bool = False
    ) -> torch.Tensor: # TODO: make the for loop faster (AKA, don't use a for loop, you dingus)
    '''
    Reorder the noise tensor to have pairings for optimal transport with respect to the latent tensor.
    returns the noise tensor reordered.
    '''
    B, C, L = latent.shape

    xs = latent.view(B, -1)
    xt = noise.view(B, -1)

    a, b = torch.ones((B,)) / B, torch.ones((B,)) / B  # uniform distribution on samples

    # loss matrix
    M = ot.dist(xs, xt)
    G0 = ot.emd(a, b, M)
    bool_g = (G0*B).to(dtype=torch.bool)

    sorted_xt = torch.zeros_like(xt)
    for i in range(B):
            for j in range(B):
                if bool_g[i, j]:
                    sorted_xt[i] = xt[j]

    noise = sorted_xt.view(B, C, L)
    return noise

def make_clustered_dataloader(batch_size: int,
                              mega_batch: int,
                              dataset: Dataset,
                              tokenizer: PreTrainedTokenizerFast,
                              max_len: int,
                              id_key: str = 'id',
                              length_key: str = 'length',
                              label_key: str = 'label',
                              sequence_key: str = 'sequence',
                              pad_to_multiple_of: int = 16,
                              drop_last: bool = False,
                              num_workers: int = 1,
                              seed: int = 42,
                              shuffle: bool = True,
                              random_padding: bool = False,
) -> DataLoader:

    sampler = BatchSampler(dataset,
                           tokenizer,
                           batch_size,
                           mega_batch,
                           max_length=max_len,
                           id_key=id_key,
                           length_key=length_key,
                           label_key=label_key,
                           sequence_key=sequence_key,
                           pad_to_multiple_of=pad_to_multiple_of,
                           drop_last=drop_last,
                           num_workers=num_workers,
                           seed=seed,
                           shuffle=shuffle,
                           random_padding=random_padding,
    )

    clustered_dataset = ClusteredDataset(dataset, 
                                        id_key=id_key,
                                        length_key=length_key,
                                        label_key=label_key,
                                        sequence_key=sequence_key,
                                        pad_to_multiple_of=pad_to_multiple_of,
    )

    dataloader = DataLoader(clustered_dataset,
                            batch_sampler=sampler, 
                            collate_fn=sampler.collate_fn,
                            num_workers=num_workers,
    )
    return dataloader

# def make_normal_dataloader(config: VAETrainingConfig,
#                         dataset: Dataset,
#                         tokenizer: PreTrainedTokenizerFast,
#                         max_len: int,
#                         id_key: str = 'id',
#                         length_key: str = 'length',
#                         label_key: str = 'label',
#                         sequence_key: str = 'sequence',
#                         pad_to_multiple_of: int = 16,
#                         drop_last: bool = False,
#                         num_workers: int = 1,
#                         generator: Optional[torch.Generator] = None,
# ) -> DataLoader:

#     sampler = BatchSampler(dataset,
#                            tokenizer,
#                            config.batch_size,
#                            config.mega_batch,
#                            max_length=max_len,
#                            id_key=id_key,
#                            length_key=length_key,
#                            label_key=label_key,
#                            sequence_key=sequence_key,
#                            pad_to_multiple_of=pad_to_multiple_of,
#                            drop_last=drop_last,
#                            num_workers=num_workers,
#                            generator=generator,
#     )

#     clustered_dataset = ClusteredDataset(dataset, 
#                                          id_key=id_key,
#                                          length_key=length_key,
#                                          label_key=label_key,
#                                          sequence_key=sequence_key,
#                                          pad_to_multiple_of=pad_to_multiple_of,
#     )

#     dataloader = DataLoader(clustered_dataset,
#                             batch_sampler=sampler, 
#                             collate_fn=sampler.collate_fn,
#                             num_workers=num_workers,
#                             generator=generator,
#     )
#     return dataloader

class ClusteredDataset(Dataset):
    '''
    Create a custom dataset for the clustered dataset.
    The dataset is a dictionary with the identifier as the key, and the value is a dictionary with the label, list of sequences, and list of lengths.
    '''
    def __init__(self, dataset, 
                 id_key: str = 'id',
                 length_key: str = 'lengths',
                 label_key: str = 'label',
                 sequence_key: str = 'sequence',
                 pad_to_multiple_of: int = 16,
    ):
        self.dataset = dataset
        self.id_key = id_key
        self.length_key = length_key
        self.label_key = label_key
        self.sequence_key = sequence_key
        self.pad_to_multiple_of = pad_to_multiple_of

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: Union[List[List[int]], torch.Tensor]): # Way too convoluted, I'm sorry.
        '''
        Get a sample from the dataset. Using two indices, the first index is the cluster index, and the second index is the sample index.
        
        If you choose a single index, or a list of single integers it will return the entire cluster.
        '''

        if torch.is_tensor(idx):
            idx = idx.detach().tolist()

        assert isinstance(idx, List), "The index must be a list"
        assert all(isinstance(i, List) for i in idx), "The index must be a list of lists"
        assert all(len(i) == 2 for i in idx), "The index must be a list of lists with two elements"
        assert all(isinstance(i, int) for pair in idx for i in pair), "The elements of the index must be integers"

        clusterindex = [pair[0] for pair in idx]
        sampleindex = [pair[1] for pair in idx]

        data = self.dataset[clusterindex]

        id = data[self.id_key]
        length = []
        label = []
        sequence = []

        for i in range(len(idx)):
            sampleindex_i = sampleindex[i]
            length.append(data[self.length_key][i][sampleindex_i])
            label.append(data[self.label_key][i][sampleindex_i])
            sequence.append(data[self.sequence_key][i][sampleindex_i])
        
        id = np.array(id).astype(np.bytes_)
        sequence = np.array(sequence).astype(np.bytes_)
        label = np.array(label).astype(np.int64)
        length = np.array(length).astype(np.int64)

        return {'id': id, 'length': length, 'label': label, 'sequence': sequence}

class BatchSampler(Sampler): 
    '''
    BatchSampler for variable length sequences, batching by similar lengths, to prevent excessive padding.
    '''
    def __init__(self, 
                 dataset: Dataset, 
                 tokenizer: PreTrainedTokenizerFast,
                 batch_size: int, 
                 mega_batch_size: int, 
                 max_length: Optional[int] = None,
                 id_key: str = 'id',
                 length_key: str = 'length',
                 label_key: str = 'label',
                 sequence_key: str = 'input_ids',
                 pad_to_multiple_of: int = 8,
                 drop_last: bool = True,
                 num_workers: int = 1,
                 seed: int = 42,
                 shuffle: bool = True,
                 random_padding: bool = False, # wheter to pad randomly on either side, or just right padding
    ):
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.mega_batch_size = mega_batch_size
        self.drop_last = drop_last
        self.max_length = max_length
        self.id_key = id_key
        self.length_key = length_key
        self.label_key = label_key
        self.sequence_key = sequence_key
        self.pad_to_multiple_of = pad_to_multiple_of
        self.dataset = dataset
        self.num_workers = num_workers
        self.seed = seed
        self.random_padding = random_padding
        if shuffle:
            self.generator = torch.Generator().manual_seed(seed)
        else:
            self.generator = None
        self.shuffle = shuffle
        assert self.batch_size * self.mega_batch_size // self.num_workers > 0, "The batch size times the mega batch size must be larger than the number of workers."

    def collate_fn(self, batch):
        '''
        Collate function for the DataLoader.
        Takes dictionary with the keys id_key, length_key, label_key, and sequence_key.
        The length key has to be already rounded to the amount the sequence will be padded to.
        This has to be done as this value is also needed in the BatchSampler, so th elength has to be precomputed when making the dataset. See convert_csv_to_dataset.py for an example.
        The value of each key is a list.
        Returns a dictionary with the keys 'id', 'label', and 'sequence'.
        '''

        # assert all(item[self.length_key] % self.pad_to_multiple_of == 0 for item in batch), "The length_key values of the sequences must be a multiple of the pad_to_multiple_of parameter." #TODO: Could be commented out and made an assertion on the dataset level.

        length_list = [round_length(item[self.length_key], pad = 2, rounding = self.pad_to_multiple_of) for item in batch]
        sample_max_len = max(length_list)
        max_length_cap = self.max_length

        if max_length_cap is not None:
            max_len = min(sample_max_len, max_length_cap)
        else:
            max_len = sample_max_len

        id_list = []
        seq_list = []
        label_list = []

        for i, item in enumerate(batch):
            # id
            id = item[self.id_key]
            id = str(id, encoding = 'utf-8')
            id_list.append(id)

            # label
            label_list.append(item[self.label_key])

            # sequence
            seq = item[self.sequence_key]
            seq = str(seq, encoding='utf-8')
            seq = process_sequence(seq, pad_to_multiple_of=self.pad_to_multiple_of, generator=self.generator, random_padding=self.random_padding)
            seq_len = length_list[i] # make sure to get the processed seq_length - meaning what it is after tokenization

            if seq_len > max_len:
                index = torch.randint(0, seq_len - max_len, (1,), generator=self.generator).item()
                seq_list.append(seq[index:index+max_len])
            else:
                seq_list.append(seq)

        tokenized = tokenize_sequence(seq_list, tokenizer=self.tokenizer)
        
        input_ids = tokenized['input_ids']
        attention_mask: torch.BoolTensor = (input_ids != self.tokenizer.pad_token_id).to(dtype=torch.bool) # Attention mask should be bool for scaled_dot_product_attention
        # attention_mask = tokenized['attention_mask'].to(dtype=torch.bool) # Attention mask should be bool for scaled_dot_product_attention
        label = torch.tensor(label_list)
        length = torch.tensor(length_list)
        id = np.array(id_list).astype(np.bytes_)
        sequence = np.array(seq_list).astype(np.bytes_)

        # label = label_list
        # length = length_list
        # id = id_list
        # sequence = seq_list

        return {
            'id': id, 
            'label': label,
            # 'sequence': sequence,
            'length': length,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
        }

    def get_lengths(self, indices, return_dict, process_id):
        return_dict[process_id] = self.dataset[indices][self.length_key]

    def __iter__(self):

        # If shuffle is false, we remake the generator each epoch for deterministic val loaders
        if self.shuffle == False:
            self.generator = torch.Generator().manual_seed(self.seed)

        size = len(self.dataset)

        indices = torch.randperm(size, generator=self.generator)

        step = self.mega_batch_size * self.batch_size
        for i in range(0, size, step):
            pool_indices = indices[i:i+step]

            # New implementation
            # Use torch.multiprocessing to get lengths
            with mp.Manager() as manager:
                return_dict = manager.dict()
                processes = []
                chunk_size = step // self.num_workers
                chunks = [pool_indices[j:j + chunk_size] for j in range(0, step, chunk_size)]

                for process_id, chunk in enumerate(chunks):
                    p = mp.Process(target=self.get_lengths, args=(chunk, return_dict, process_id))
                    p.start()
                    processes.append(p)

                for p in processes:
                    p.join()
                
                # Retrieve results in order
                pool_lenlists = [lenlist for process_id in sorted(return_dict.keys()) for lenlist in return_dict[process_id]]

            # # New implementation
            lengths = torch.zeros((len(pool_indices),), dtype=torch.int64) # lengths of the sequences
            sample_indices = -1 * torch.ones((len(pool_indices),), dtype=torch.int64) # the indices of the samples in the clusters

            for i, lenlist in enumerate(pool_lenlists):
                sample_randint = torch.randint(0, len(lenlist), (1,), generator=self.generator).item()
                
                # Set the lengths
                lengths[i] = lenlist[sample_randint]

                # Set the indices
                sample_indices[i] = sample_randint

            # Sort the lengths
            sorted_lengths, sorted_indices = torch.sort(lengths)

            # Use the sorted indices to sort the pool_indices and sample_indices by the lengths retrieved
            pool_sample_indices = torch.cat((pool_indices.unsqueeze(1), sample_indices.unsqueeze(1)), dim=1) # len(pool_indices) * (cluster_id, sample_id)
            pool = pool_sample_indices[sorted_indices] # of shape (len(pool_indices), 2)

            # # Old implementation
            # sample_indices = [torch.randint(0, len(lenlist), (1,), generator=self.generator) for lenlist in pool_lengths]
            # pool_lengths = [lenlist[index] for lenlist, index in zip(pool_lengths, sample_indices)] # list of lengths

            # # Zip indices with sample indices and sort by length more efficiently
            # pool = list(zip(pool_lengths, pool_indices, sample_indices))
            # pool.sort(key=lambda x: x[0])

            # pool = [(pool_id, sample_id) for _, pool_id, sample_id in pool]

            mega_batch_indices = torch.randperm(math.ceil(pool.shape[0] / self.batch_size), generator=self.generator) * self.batch_size

            for j in mega_batch_indices:
                if self.drop_last and j + self.batch_size > pool.shape[0]: # drop the last batch if it's too small
                    continue

                batch = pool[j:j+self.batch_size] # (pool_id, sample_id). First is the id of the cluster, second is the id of the sample in the cluster.

                yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        else:
            return (len(self.dataset) + self.batch_size - 1) // self.batch_size

@dataclass
class TrainingVariables:
    global_step: int = 0
    val_loss: float = float("inf")
    max_len_start: int = 0
    n_tokens: int = 0

    def state_dict(self):
        return self.__dict__
    
    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)

class VAETrainer:
    def __init__(self, 
                 model: AutoencoderKL1D, 
                 tokenizer: PreTrainedTokenizerFast, 
                 train_dataloader: DataLoader, 
                 val_dataloader: DataLoader, 
                 config: VAETrainingConfig, 
                 test_dataloader: DataLoader = None,
                 training_variables: Optional[TrainingVariables] = None,
        ):
        self.tokenizer = tokenizer
        self.config = config
        self.training_variables = training_variables or TrainingVariables()
        self.accelerator_config = ProjectConfiguration(
            project_dir=self.config.output_dir,
            logging_dir=os.path.join(self.config.output_dir, "logs"),
            automatic_checkpoint_naming=self.config.automatic_checkpoint_naming,
            total_limit=self.config.total_checkpoints_limit, # Limit the total number of checkpoints
        )
        self.accelerator = Accelerator(
            project_config=self.accelerator_config,
            mixed_precision=self.config.mixed_precision,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            log_with="tensorboard",
        )

        if config.optimizer == "Adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        elif config.optimizer == "AdamW":
            optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        elif config.optimizer == "Adamax":
            optimizer = torch.optim.Adamax(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        elif config.optimizer == "SGD":
            assert config.SGDmomentum is not None, "SGD requires a momentum value"
            optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay, momentum=config.SGDmomentum)
        else:
            raise ValueError("Invalid optimizer, choose between `AdamW`, `Adam`, `SGD`, and `Adamax`")

        num_training_steps = (len(train_dataloader) * config.num_epochs) // self.config.gradient_accumulation_steps

        if config.lr_schedule == 'constant':
            lr_scheduler = get_constant_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=config.lr_warmup_steps,
            )
        elif config.lr_schedule == 'cosine':
            lr_scheduler = get_cosine_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=config.lr_warmup_steps,
                num_training_steps=num_training_steps,
            )
        elif config.lr_schedule == 'cosine_10x_decay':
            lr_scheduler = get_cosine_10x_decay_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=config.lr_warmup_steps,
                num_training_steps=num_training_steps,
            )
        elif config.lr_schedule == 'cosine_100x_decay':
            lr_scheduler = get_cosine_100x_decay_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=config.lr_warmup_steps,
                num_training_steps=num_training_steps,
            )
        else:
            raise NotImplementedError('unknown lr schedule: {config.lr_schedule}')

        # Create the output directory
        if self.accelerator.is_main_process:
            if not os.path.exists(self.config.output_dir):
                os.makedirs(self.config.output_dir, exist_ok=False)
            elif not self.config.overwrite_output_dir:
                raise ValueError("Output directory already exists. Set `config.overwrite_output_dir` to `True` to overwrite it.")
            else:
                raise NotImplementedError(f'Overwriting the output directory {self.config.output_dir} is not implemented yet, please delete the directory manually.')

            if self.config.push_to_hub:
                raise NotImplementedError("Pushing to the HF Hub is not implemented yet")

        # Start the logging
        self.accelerator.init_trackers(
            project_name=self.accelerator_config.logging_dir,
            config=vars(self.config),
        )

        # Prepare everything
        # There is no specific order to remember, you just need to unpack the
        # objects in the same order you gave them to the prepare method.
        ema = EMA(model, 
                  beta = self.config.ema_decay, 
                  update_after_step = self.config.ema_update_after,
                  update_every = self.config.ema_update_every
        )
        self.model, self.ema, self.optimizer, self.train_dataloader, self.test_dataloader, self.val_dataloader, self.lr_scheduler = self.accelerator.prepare(
            model, ema, optimizer, train_dataloader, test_dataloader, val_dataloader, lr_scheduler
        )
        self.accelerator.register_for_checkpointing(self.training_variables)

        # Retrieve variables set in the dataloader
        self.pad_to_multiple_of = self.train_dataloader.dataset.pad_to_multiple_of
        self.seq_key = self.train_dataloader.dataset.sequence_key
        self.label_key = self.train_dataloader.dataset.label_key

    def get_kl_weight(self):
        if self.config.kl_schedule == 'constant_with_warmup':
            progress = self.training_variables.global_step / (self.config.kl_warmup_steps * self.config.gradient_accumulation_steps)
            kl_weight = self.config.kl_weight * min(1.0, progress)
        if self.config.kl_schedule == 'constant_with_restarts':
            if self.training_variables.global_step + (2 * (self.config.kl_warmup_steps * self.config.gradient_accumulation_steps)) >= len(self.train_dataloader) * self.config.num_epochs: # If theres is less than a cycle left, set the 
                progress = 1.5
            else:
                steps = self.training_variables.global_step % (2 * (self.config.kl_warmup_steps * self.config.gradient_accumulation_steps))
                progress = steps / (self.config.kl_warmup_steps * self.config.gradient_accumulation_steps)
            kl_weight = self.config.kl_weight * min(1.0, progress)
        return kl_weight

    def update_max_len(self):
        # double the max length, max_len_start, up till the cap, max_len
        if self.training_variables.max_len_start > self.config.max_len:
            self.training_variables.max_len_start = self.config.max_len
            print(f"Somehow, the current batch max length was higher than the cap, setting it to the cap. This should not be able to happen, but I wont throw an error.")
        if self.training_variables.max_len_start < self.config.max_len:
            self.training_variables.max_len_start = min(self.training_variables.max_len_start * 2, self.config.max_len) # To not have an int exploding to infinity in the background
            print(f"Updating the batch max length to {self.training_variables.max_len_start}")
        
        self.train_dataloader.batch_sampler.max_length = self.training_variables.max_len_start


    def scale_gradients(self, 
                        m: nn.Module, 
                        n_tokens: int = 1, 
    ):
        '''
        Scale the gradients by dividing by the amount of tokens. Necessary for gradient accumulation with different length batches.
        '''

        for p in m.parameters():
            if p.requires_grad and p.grad is not None:
                p.grad.data = p.grad.data / n_tokens

    @torch.no_grad()
    def evaluate(self,
                 model: AutoencoderKL1D,
    ) -> dict:
        model.eval()
        test_dir = os.path.join(self.config.output_dir, "samples")
        os.makedirs(test_dir, exist_ok=True)

        running_loss = 0.0
        running_loss_ce = 0.0
        running_loss_kl = 0.0
        num_correct_residues = 0
        total_residues = 0
        latent_data = []
        name = f"step_{self.training_variables.global_step//1:08d}"

        progress_bar = tqdm(total=len(self.val_dataloader), disable = True) # not self.accelerator.is_local_main_process)
        progress_bar.set_description(f"Evaluating {name}")

        for i, batch in enumerate(self.val_dataloader):

            input_ids: torch.IntTensor = batch['input_ids']
            attention_mask: torch.BoolTensor = batch['attention_mask']
            length: torch.IntTensor = batch['length']

            output = model(sample = input_ids,
                           attention_mask = attention_mask,
                           sample_posterior = True, # Could be set to False in inference. This will take the mode of the latent dist TODO: should sample_posterior be true or false?
            )

            ce_loss, kl_loss = model.loss_fn(output, input_ids)
            loss = ce_loss + kl_loss * self.config.kl_weight
            running_loss_ce += ce_loss.detach().item()
            running_loss_kl += kl_loss.detach().item()
            running_loss += loss.detach().item()

            if i == 0 and self.accelerator.is_main_process: # save the first sample each evaluation as a logoplot
                logoplot_sample = output.sample[0]
                # remove the padding
                logoplot_sample_len = length[0]
                logoplot_sample = logoplot_sample[:,:logoplot_sample_len]
                logoplot_sample_id = str(batch['id'][0], encoding='utf-8')
                probs = F.softmax(logoplot_sample, dim=0).cpu().numpy()
                pool = mp.Pool(1)
                pool.apply_async(make_logoplot, 
                                args=(
                                    probs, 
                                    logoplot_sample_id, 
                                    f"{test_dir}/{name}_probs_{logoplot_sample_id}.png", 
                                    self.tokenizer.decode(range(self.tokenizer.vocab_size)),
                                ),
                                error_callback=lambda e: print(e),
                                callback=lambda _: pool.close(),
                )
                gc.collect()

            token_ids_pred = logits_to_token_ids(output.sample, self.tokenizer, cutoff=self.config.cutoff)

            token_ids_correct = ((input_ids == token_ids_pred) & (attention_mask == 1)).long()
            num_residues = torch.sum(attention_mask, dim=1).long()

            num_correct_residues += token_ids_correct.detach().sum().item()
            total_residues += num_residues.detach().sum().item()

            # get the latent space data
            latent: torch.Tensor = output.latent_dist.sample() # TODO: sample or mode?
            mask: torch.BoolTensor = output.attention_masks[-1].unsqueeze(1).expand_as(latent)
            data = latent[mask].detach().tolist()
            latent_data.extend(data)

            # Decode the predicted sequences, and remove zero padding
            seqs_pred = self.tokenizer.batch_decode(token_ids_pred, skip_special_tokens=self.config.skip_special_tokens)
            seqs_lens = length

            # Remove the padding from the sequences
            seqs_pred = [seq[:i] for seq, i in zip(seqs_pred, seqs_lens)]

            # Save all samples as a FASTA file
            seq_record_list = [SeqRecord(Seq(seq), id=str(batch['id'][i], encoding='utf-8'), 
            # seq_record_list = [SeqRecord(Seq(seq), id=str(batch['id'][i]), 
                            description=
                            f"label: {batch[self.label_key][i]} acc: {token_ids_correct[i].sum().item() / num_residues[i].item():.3f}")
                            for i, seq in enumerate(seqs_pred)]

            with open(f"{test_dir}/{name}.fa", "a") as f:
                SeqIO.write(seq_record_list, f, "fasta")

            progress_bar.update(1)

        # Calculate statistics for the validation set
        mu = np.mean(latent_data)
        std = np.std(latent_data)
        model.config.scaling_factor = 1/std # Set the scaling factor for the VAE to be 1/std, so that the latent space is normalized sd=1

        acc = num_correct_residues / total_residues
        log_loss = running_loss / len(self.val_dataloader)
        log_loss_ce = running_loss_ce / len(self.val_dataloader)
        log_loss_kl = running_loss_kl / len(self.val_dataloader)
        print(f"{name}, val_loss: {log_loss:.4f}, val_accuracy: {acc:.4f}, val_mu: {mu:.4f}, val_std: {std:.4f}")
        logs = {"val_loss": log_loss, 
                "val_ce_loss": log_loss_ce, 
                "val_kl_loss": log_loss_kl,
                "val_acc": acc,
                "val_mu": mu,
                "val_std": std,
                }
        gc.collect()
        return logs

    def train(self, from_checkpoint: Optional[Union[str, os.PathLike]] = None):

        # start the loop
        if self.accelerator.is_local_main_process:
            # load the checkpoint if it exists
            if from_checkpoint is None:
                skipped_dataloader = self.train_dataloader
                starting_epoch = 0
                self.training_variables.global_step = 0
                self.training_variables.val_loss = float("inf")
                self.training_variables.max_len_start = self.config.max_len_start
                self.training_variables.n_tokens = 0
            else:
                self.accelerator.load_state(input_dir=from_checkpoint)
                # self.lr_scheduler._initial_step()
                # for step in self.training_variables.global_step:
                #     self.lr_scheduler.step()
                # # Skip the first batches
                starting_epoch = self.training_variables.global_step // len(self.train_dataloader)
                batches_to_skip = self.training_variables.global_step % len(self.train_dataloader)
                skipped_dataloader = self.accelerator.skip_first_batches(self.train_dataloader, batches_to_skip)
                self.accelerator.print(f"Loaded checkpoint from {from_checkpoint}")
                self.accelerator.print(f"Starting from epoch {starting_epoch}")
                self.accelerator.print(f"Starting from step {self.training_variables.global_step}")
                self.accelerator.print(f"Skipping {batches_to_skip} batches (randomly)")
                self.accelerator.print(f"Current validation loss: {self.training_variables.val_loss}")
                self.accelerator.print(f"Current max length: {self.training_variables.max_len_start}")
                print(f"max length before: {self.train_dataloader.batch_sampler.max_length}")
                self.train_dataloader.batch_sampler.max_length = self.training_variables.max_len_start # Important! this should be refactored for clarity, but for now, I'll update the max len here.
                print(f"max length after: {self.train_dataloader.batch_sampler.max_length}")

        # Now you train the model
        self.model.train()

        for epoch in range(starting_epoch, self.config.num_epochs):

            if epoch == starting_epoch:
                dataloader = skipped_dataloader
            else:
                dataloader = self.train_dataloader

            progress_bar = tqdm(total=len(dataloader), disable = True) #not self.accelerator.is_local_main_process)
            progress_bar.set_description(f"Epoch {epoch}")

            for step, batch in enumerate(dataloader):

                # Gradient accumulation
                with self.accelerator.accumulate(self.model):

                    input_ids: torch.IntTensor = batch['input_ids']
                    attention_mask: torch.BoolTensor = batch['attention_mask']

                    self.training_variables.n_tokens += attention_mask.sum()

                    # Forward pass
                    output: AutoencoderKLOutput1D = self.model(sample = input_ids,
                                        attention_mask = attention_mask,
                                        sample_posterior = True, # Should be set to true in training
                    )

                    # Loss calculation
                    ce_loss, kl_loss = self.model.loss_fn(output, input_ids)
                    kl_weight = self.get_kl_weight()
                    loss = ce_loss + kl_loss * kl_weight
                    loss_back = loss * attention_mask.sum() # https://www.reddit.com/r/MachineLearning/comments/1acbzrx/d_gradient_accumulation_should_not_be_used_with/

                    # Backward pass
                    self.accelerator.backward(loss_back)

                    # Gradient clipping and gradient scaling for gradient accumulation with different length batches
                    if self.accelerator.sync_gradients:
                        self.scale_gradients(self.model, self.training_variables.n_tokens)
                        self.training_variables.n_tokens = 0
                        if self.config.gradient_clip_val is not None:
                            self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_val)

                    # Update the model
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                    if self.accelerator.sync_gradients:
                        self.ema.update()

                # Log the progress
                progress_bar.update(1)
                logs = {"train_loss": loss.detach().item(), 
                        "train_ce_loss": ce_loss.detach().item(), 
                        "train_kl_loss": kl_loss.detach().item(), 
                        "kl_weight": kl_weight,
                        "lr": self.lr_scheduler.get_last_lr()[0], 
                        "step": self.training_variables.global_step,
                }
                progress_bar.set_postfix(**logs)
                self.accelerator.log(logs, step=self.training_variables.global_step)
                self.training_variables.global_step += 1

                # Update the max_len
                if self.training_variables.global_step % self.config.max_len_doubling_steps == 0:
                    self.update_max_len()

                if self.training_variables.global_step == 1 or self.training_variables.global_step % self.config.save_image_model_steps == 0 or self.training_variables.global_step == len(self.train_dataloader) * self.config.num_epochs:
                    
                    logs = self.evaluate(self.ema.ema_model)
                    self.accelerator.log(logs, step=self.training_variables.global_step)

                    new_val_loss = logs["val_loss"]

                    self.accelerator.wait_for_everyone()
                    if True: # new_val_loss < self.training_variables.val_loss: # Save the model if the validation loss is lower
                        self.training_variables.val_loss = new_val_loss
                        self.accelerator.save_state()
                    self.model.train() # Make sure the model is in train mode
                
                self.accelerator.wait_for_everyone()
                if self.training_variables.global_step % len(self.train_dataloader) == 0: # If it is the last batch of an Epoch, save the model for easy restart.
                    self.accelerator_config.automatic_checkpoint_naming = False
                    self.accelerator.save_state(
                            output_dir=os.path.join(self.config.output_dir, f"Epoch_{epoch}")
                        )
                    self.accelerator_config.automatic_checkpoint_naming = True

        self.accelerator.end_training()
        self.save_pretrained()

    def save_pretrained(self, output_dir: Optional[str] = None):

        ce_model = self.accelerator.unwrap_model(self.model)
        ema_model = self.accelerator.unwrap_model(self.ema.ema_model)

        if output_dir is None:
            output_dir = os.path.join(self.config.output_dir, "pretrained")
        
        os.makedirs(output_dir, exist_ok=True)
        
        ce_model.save_pretrained(os.path.join(output_dir, "CE"))
        ema_model.save_pretrained(os.path.join(output_dir, "EMA"))

# Diffusion training
@dataclass
class ProtDiffusionTrainingConfig:
    batch_size: int = 64  # the batch size
    mega_batch: int = 1000 # how many batches to use for batchsampling
    num_epochs: int = 1  # the number of epochs to train the model
    gradient_accumulation_steps: int = 2  # the number of steps to accumulate gradients before taking an optimizer step
    learning_rate: float = 1e-4  # the learning rate
    lr_warmup_steps: int  = 1000
    lr_schedule: str = 'cosine'
    save_image_model_steps: int  = 1000
    save_epochs: int = 1  # How often to save the model
    mixed_precision: str = "fp16"  # `no` for float32, `fp16` for automatic mixed precision #TODO: implement fully
    optimizer: str = "AdamW"  # the optimizer to use, choose between `AdamW`, `Adam`, `SGD`, and `Adamax`
    SGDmomentum: Optional[float] = 0.9
    output_dir: str = os.path.join("output","test")  # the model name locally and on the HF Hub
    pad_to_multiple_of: int = 16 # should be a multiple of 2 for each layer in the VAE.
    max_len: int = 512  # truncation of the input sequence
    max_len_start: Optional[int] = 64  # the starting length of the input sequence
    max_len_doubling_steps: Optional[int] = 100000  # the number of steps to double the input sequence length

    class_embeddings_concat = False  # whether to concatenate the class embeddings to the time embeddings

    push_to_hub = False  # Not implemented yet. Whether to upload the saved model to the HF Hub
    hub_model_id = "kkj15dk/ProtDiffusion"  # the name of the repository to create on the HF Hub
    hub_private_repo = False
    overwrite_output_dir = False  # overwrite the old model when re-running the notebook
    seed: int = 42

    automatic_checkpoint_naming: bool = True  # whether to automatically name the checkpoints
    total_checkpoints_limit: int = 5  # the total limit of checkpoints to save

    cutoff: Optional[float] = None # cutoff for when to predict the token given the logits, and when to assign the unknown token 'X' to this position
    skip_special_tokens: bool = False # whether to skip the special tokens when writing the evaluation sequences

    gradient_clip_val: Optional[float] = 5.0  # the value to clip the gradients to
    weight_decay: float = 0.01 # weight decay for the optimizer
    ema_decay: float = 0.9999 # the decay rate for the EMA
    ema_update_after: int = 1000 # the number of steps to wait before updating the EMA
    ema_update_every: int = 1 # the number of steps to wait before updating the EMA

    use_batch_optimal_transport: bool = True # whether to use optimal transport for the batch to reorder the noise
    use_logitnorm_timestep_sampling: bool = False # whether to use the logitnorm sampling for the timestep
    logitnorm_m: float = 0 # the m parameter for the logitnorm sampling
    logitnorm_s: float = 1 # the s parameter for the logitnorm sampling

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

        if not self.overwrite_output_dir and os.path.exists(self.output_dir):
            raise ValueError("Output directory already exists. Set `config.overwrite_output_dir` to `True` to overwrite it.")

        if self.push_to_hub:
            raise NotImplementedError("Pushing to the HF Hub is not implemented yet")

        assert self.optimizer in ["AdamW", "Adam", "SGD", "Adamax"], "Invalid optimizer, choose between `AdamW`, `Adam`, `SGD`, and `Adamax`"
        assert self.mixed_precision in ["no", "fp16"], "Invalid mixed precision setting, choose between `no` and `fp16`" # TODO: implement fully
        assert self.max_len % self.pad_to_multiple_of == 0, "The maximum length of the input sequence must be a multiple of the pad_to_multiple_of parameter."
        assert self.max_len_start is None or self.max_len_start % self.pad_to_multiple_of == 0, "The starting length of the input sequence must be a multiple of the pad_to_multiple_of parameter."

        if self.max_len_start is not None:
            assert self.max_len_start <= self.max_len, "The starting length of the input sequence must be less than or equal to the maximum length of the input sequence, or None."

class ProtDiffusionTrainer:
    def __init__(self, 
                 transformer: DiTTransformer1DModel,
                 vae: AutoencoderKL1D, 
                 tokenizer: PreTrainedTokenizerFast, 
                 config: ProtDiffusionTrainingConfig, 
                 noise_scheduler: Union[DDPMScheduler, FlowMatchingEulerScheduler], # the scheduler to use for the diffusion
                 train_dataloader: DataLoader, 
                 val_dataloader: Optional[DataLoader] = None, 
                 test_dataloader: Optional[DataLoader] = None,
                 training_variables: Optional[TrainingVariables] = None,
                 eval_seq_len: Union[List[int],int] = 1024, # the sequence lengths to evaluate on
                 eval_class_labels: Optional[Union[List[int],int]] = None, # the class labels to evaluate on, should be a list the same length as the eval batch size
                 eval_guidance_scale: float = 2.0, # the scale of the guidance for the diffusion
                 eval_num_inference_steps: int = 1000, # the number of inference steps for the diffusion
        ):
        self.noise_scheduler: Union[DDPMScheduler, FlowMatchingEulerScheduler] = noise_scheduler
        self.transformer: DiTTransformer1DModel
        self.vae: AutoencoderKL1D
        self.tokenizer: PreTrainedTokenizerFast
        self.optimizer: torch.optim.Optimizer

        assert isinstance(noise_scheduler, (DDPMScheduler, FlowMatchingEulerScheduler)), "The noise scheduler must be an instance of KarrasDiffusionSchedulers or FlowMatchingEulerScheduler"
        # Figure out if we are doing flow (matching) or diffusion
        if isinstance(noise_scheduler, DDPMScheduler):
            self.flow = False
            self.diffusion = True
        elif isinstance(noise_scheduler, FlowMatchingEulerScheduler):
            self.flow = True
            self.diffusion = False
        self.tokenizer = tokenizer
        self.config = config

        self.eval_seq_len = eval_seq_len
        assert isinstance(eval_seq_len, (int, list)), "The evaluation sequence length should be an integer or a list of integers"
        assert all(seq_len <= self.config.max_len for seq_len in eval_seq_len), "The evaluation sequence length must not be greater than the maximum sequence length"

        if vae.config.scaling_factor is None:
            print("No scaling factor found in the VAE, setting it to 1.0")
            self.scaling_factor = 1.0
        else:
            print(f"Scaling factor found in the VAE: {vae.config.scaling_factor}")
            self.scaling_factor = vae.config.scaling_factor

        self.eval_class_labels = eval_class_labels
        self.eval_guidance_scale = eval_guidance_scale
        self.eval_num_inference_steps = eval_num_inference_steps
        self.training_variables = training_variables or TrainingVariables()
        self.accelerator_config = ProjectConfiguration(
            project_dir=self.config.output_dir,
            logging_dir=os.path.join(self.config.output_dir, "logs"),
            automatic_checkpoint_naming=self.config.automatic_checkpoint_naming,
            total_limit=self.config.total_checkpoints_limit, # Limit the total number of checkpoints
        )
        self.accelerator = Accelerator(
            project_config=self.accelerator_config,
            mixed_precision=self.config.mixed_precision,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            log_with="tensorboard",
        )

        if config.optimizer == "Adam":
            optimizer = torch.optim.Adam(transformer.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        elif config.optimizer == "AdamW":
            optimizer = torch.optim.AdamW(transformer.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        elif config.optimizer == "Adamax":
            optimizer = torch.optim.Adamax(transformer.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        elif config.optimizer == "SGD":
            assert config.SGDmomentum is not None, "SGD requires a momentum value"
            optimizer = torch.optim.SGD(transformer.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay, momentum=config.SGDmomentum)
        else:
            raise ValueError("Invalid optimizer, choose between `AdamW`, `Adam`, `SGD`, and `Adamax`")

        num_training_steps = (len(train_dataloader) * config.num_epochs) // self.config.gradient_accumulation_steps

        if config.lr_schedule == 'constant':
            lr_scheduler = get_constant_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=config.lr_warmup_steps,
            )
        elif config.lr_schedule == 'cosine':
            lr_scheduler = get_cosine_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=config.lr_warmup_steps,
                num_training_steps=num_training_steps,
            )
        elif config.lr_schedule == 'cosine_10x_decay':
            lr_scheduler = get_cosine_10x_decay_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=config.lr_warmup_steps,
                num_training_steps=num_training_steps,
            )
        elif config.lr_schedule == 'cosine_100x_decay':
            lr_scheduler = get_cosine_100x_decay_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=config.lr_warmup_steps,
                num_training_steps=num_training_steps,
            )
        else:
            raise NotImplementedError('unknown lr schedule: {config.lr_schedule}')
        
        # Create the output directory
        if self.accelerator.is_main_process:
            if not os.path.exists(self.config.output_dir):
                os.makedirs(self.config.output_dir, exist_ok=False)
            elif not self.config.overwrite_output_dir:
                raise ValueError("Output directory already exists. Set `config.overwrite_output_dir` to `True` to overwrite it.")
            else:
                raise NotImplementedError(f'Overwriting the output directory {self.config.output_dir} is not implemented yet, please delete the directory manually.')
                
            if self.config.push_to_hub:
                raise NotImplementedError("Pushing to the HF Hub is not implemented yet")
        
        # Start the logging
        self.accelerator.init_trackers(
            project_name=self.accelerator_config.logging_dir,
            config=vars(self.config),
        )
        
        # Prepare everything
        # There is no specific order to remember, you just need to unpack the
        # objects in the same order you gave them to the prepare method.
        ema = EMA(transformer, 
                  beta = self.config.ema_decay, 
                  update_after_step = self.config.ema_update_after,
                  update_every = self.config.ema_update_every
        )
        self.transformer, self.ema, self.vae, self.optimizer, self.train_dataloader, self.lr_scheduler = self.accelerator.prepare(
            transformer, ema, vae, optimizer, train_dataloader, lr_scheduler
        )
        if test_dataloader is not None:
            self.test_dataloader: DataLoader = self.accelerator.prepare(test_dataloader)
        if val_dataloader is not None:
            self.val_dataloader: DataLoader = self.accelerator.prepare(val_dataloader)
        self.vae.eval() # Set the VAE to eval mode
        self.accelerator.register_for_checkpointing(self.training_variables)

        # Retrieve variables set in the dataloader
        self.pad_to_multiple_of = self.train_dataloader.dataset.pad_to_multiple_of
        self.seq_key = self.train_dataloader.dataset.sequence_key
        self.label_key = self.train_dataloader.dataset.label_key

    def update_max_len(self):
        # double the max length, max_len_start, up till the cap, max_len
        if self.training_variables.max_len_start > self.config.max_len:
            self.training_variables.max_len_start = self.config.max_len
            print(f"Somehow, the current batch max length was higher than the cap, setting it to the cap. This should not be able to happen, but I wont throw an error.")
        if self.training_variables.max_len_start < self.config.max_len:
            self.training_variables.max_len_start = min(self.training_variables.max_len_start * 2, self.config.max_len) # To not have an int exploding to infinity in the background
            print(f"Updating the batch max length to {self.training_variables.max_len_start}")
        
        self.train_dataloader.batch_sampler.max_length = self.training_variables.max_len_start

    def scale_gradients(self, 
                        m: nn.Module, 
                        n_tokens: int = 1, 
    ):
        '''
        Scale the gradients by dividing by the amount of tokens. Necessary for gradient accumulation with different length batches.
        '''

        for p in m.parameters():
            if p.requires_grad and p.grad is not None:
                p.grad.data = p.grad.data / n_tokens

    def logitnorm(self, bs: int, m: int, s: int, generator: Optional[torch.Generator] = None) -> torch.Tensor: # TODO: fix bug. This is wrongfully implemented
        '''
        draw random samples from a logitnormal distribution. https://arxiv.org/pdf/2403.03206
        '''

        assert m == 0 and s == 1, "The logitnormal distribution is only implemented for m=0 and s=1"

        rand = torch.randn((bs,), device=self.accelerator.device, generator=generator)
        samples = 1 / (1 + torch.exp(-rand))
        return samples

    def sample_timesteps(self, bs: int, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        '''
        Get the timesteps for the diffusion or flow matching training.
        bs: int, the batch size
        generator: torch.Generator, the random number generator
        '''

        if self.diffusion and not self.flow: # Diffusion
            if self.config.use_logitnorm_timestep_sampling:
                timesteps = self.logitnorm(bs, self.config.logitnorm_m, self.config.logitnorm_s, generator)
                timesteps = torch.round(timesteps * self.noise_scheduler.config.num_train_timesteps).to(dtype=torch.int) # scale to the number of timesteps
            timesteps = torch.randint(
                    0, self.noise_scheduler.config.num_train_timesteps, (bs,), device=self.accelerator.device,
                    dtype=torch.int64,
                    generator=generator,
            )
        elif self.flow and not self.diffusion: # Flow matching
            if self.config.use_logitnorm_timestep_sampling:
                timesteps = self.logitnorm(bs, self.config.logitnorm_m, self.config.logitnorm_s, generator)
            else:
                timesteps = torch.rand(
                        (bs,), 
                        device=self.accelerator.device,
                        dtype=torch.float32,
                        generator=generator,
                )
        return timesteps

    def loss(self,
             output: Transformer1DModelOutput,
             latent: Optional[torch.Tensor],
             noise: torch.Tensor,
             attention_mask: torch.Tensor,
    ):
        if self.diffusion and not self.flow: # Diffusion
            target = noise
        elif self.flow and not self.diffusion: # Flow matching
            target = latent - noise
        else:
            raise ValueError("Either diffusion or flow must be True, and the other False")
        loss = F.mse_loss(output, target, reduction='none').mean(dim=1) * attention_mask # Mean over the channel dimension, Mask the loss
        loss = loss.sum() / attention_mask.sum() # Average the loss over the non-masked tokens

        return loss

    def evaluate(self, model: DiTTransformer1DModel):
        model.eval()
        dataloader = self.val_dataloader
        if dataloader.batch_sampler.shuffle == False: # TODO: there's a bug in the very first evaluation. All subsequent evaluations work fine when using shuffle=False.
            generator = torch.Generator(device=self.accelerator.device).manual_seed(self.config.seed)
        else:
            generator = None

        progress_bar = tqdm(total=len(dataloader), disable = True) # not self.accelerator.is_local_main_process)

        running_loss = 0.0

        for step, batch in enumerate(dataloader):
            input_ids = batch['input_ids']
            # if step == 0:
            #     print(input_ids[0])
            #     print(batch['id'][0])
            attention_mask = batch['attention_mask']

            vae_encoded = self.vae.encode(x = input_ids,
                                          attention_mask = attention_mask,
            )

            attention_mask = vae_encoded.attention_masks[-1]
            latent = vae_encoded.latent_dist.sample(generator=generator) # Mode is deterministic TODO: .sample() or .mode()?
            latent = latent * self.scaling_factor # Scale the latent space to have sd=1

            label = batch['label']

            # Sample noise to add to the images
            noise = torch.randn(
                latent.shape, 
                device=self.accelerator.device,
                generator=generator,
            )

            if self.config.use_batch_optimal_transport:
                noise = reorder_noise_for_OT(latent, noise)
            
            noise = noise * attention_mask.unsqueeze(1) # Mask the noise
            bs = latent.shape[0]

            # Sample a random timestep for each latent. Make sure it is from a uniform distribution for evaluation.
            timesteps = torch.rand(
                        (bs,), 
                        device=self.accelerator.device,
                        dtype=torch.float32,
                        generator=generator,
            )
            if self.diffusion and not self.flow: # Diffusion
                timesteps = (self.noise_scheduler.config.num_train_timesteps * timesteps).to(dtype=torch.int) # round to the nearest integer

            # Add noise to the clean images according to the noise magnitude at each timestep
            noisy_latent = self.noise_scheduler.add_noise(latent, noise, timesteps) # Same method for both diffusion and flow matching
            noisy_latent = noisy_latent.to(self.accelerator.device)
            attention_mask = attention_mask.to(self.accelerator.device)

            # Forward pass
            output: torch.Tensor = model(hidden_states = noisy_latent,
                           attention_mask = attention_mask,
                           timestep = timesteps,
                           class_labels = label,
            ).sample
            
            # Loss calculation
            loss = self.loss(output, latent, noise, attention_mask)
            running_loss += loss.detach().item()

            progress_bar.update(1)

        val_loss = running_loss / len(dataloader)
        return val_loss

    @torch.no_grad()
    def inference_test(self,
                       pipeline: ProtDiffusionPipeline,
    ):
        test_dir = os.path.join(self.config.output_dir, "samples")
        os.makedirs(test_dir, exist_ok=True)

        seqs_lens = self.eval_seq_len
        class_labels = self.eval_class_labels
        name = f"step_{self.training_variables.global_step//1:08d}"

        output = pipeline(seq_len=seqs_lens,
                          class_labels=class_labels,
                          guidance_scale=self.eval_guidance_scale,
                          num_inference_steps=self.eval_num_inference_steps,
                          generator=None,
                          output_type='logits', # aa_seq
        )
        logits = output.seqs
        # seqs_pred = output.seqs

        # make a logoplot of the first sample
        logoplot_sample = logits[0]
        # remove the padding
        logoplot_sample_len = seqs_lens[0]
        logoplot_sample = logoplot_sample[:,:logoplot_sample_len]
        logoplot_sample_cl = class_labels[0]
        probs = F.softmax(logoplot_sample, dim=0).cpu().numpy()
        pool = mp.Pool(1)
        pool.apply_async(make_logoplot, 
                        args=(
                            probs, 
                            name, 
                            f"{test_dir}/{name}_length_{logoplot_sample_len}_class_label_{logoplot_sample_cl}_inference_steps_{self.eval_num_inference_steps}.png", 
                            self.tokenizer.decode(range(self.tokenizer.vocab_size)),
                        ),
                        error_callback=lambda e: print(e),
                        callback=lambda _: pool.close(),
        )
        # gc.collect()

        token_ids_pred = logits_to_token_ids(logits, self.tokenizer, cutoff=self.config.cutoff)

        # Decode the predicted sequences, and remove zero padding
        seqs_pred = self.tokenizer.batch_decode(token_ids_pred, skip_special_tokens=self.config.skip_special_tokens)

        # Remove the padding from the sequences
        seqs_pred = [seq[:i] for seq, i in zip(seqs_pred, seqs_lens)]

        # Save all samples as a FASTA file
        seq_record_list = [SeqRecord(Seq(seq), id=str(seqs_lens[i]), 
                        description=
                        f"length: {seqs_lens[i]} label: {class_labels[i]}")
                        for i, seq in enumerate(seqs_pred)]

        with open(f"{test_dir}/{name}.fa", "a") as f:
            SeqIO.write(seq_record_list, f, "fasta")
        
        gc.collect()
        print(f"Inference test done: {name}")
        return

    def train(self, from_checkpoint: Optional[Union[str, os.PathLike]] = None):

        # start the loop
        if self.accelerator.is_local_main_process:
            # load the checkpoint if it exists
            if from_checkpoint is None:
                skipped_dataloader = self.train_dataloader
                starting_epoch = 0
                self.training_variables.global_step = 0
                self.training_variables.val_loss = float("inf")
                self.training_variables.max_len_start = self.config.max_len_start
                self.training_variables.n_tokens = 0
            else:
                # print(f"Before optimizer: {self.optimizer}")
                self.accelerator.load_state(input_dir=from_checkpoint)
                # Skip the first batches
                starting_epoch = self.training_variables.global_step // len(self.train_dataloader)
                batches_to_skip = self.training_variables.global_step % len(self.train_dataloader)
                if batches_to_skip != 0:
                    skipped_dataloader = self.accelerator.skip_first_batches(self.train_dataloader, batches_to_skip)
                else:
                    skipped_dataloader = self.train_dataloader
                print(f"Loaded checkpoint from {from_checkpoint}")
                print(f"Starting from epoch {starting_epoch}")
                print(f"Starting from step {self.training_variables.global_step}")
                print(f"Skipping {batches_to_skip} batches (randomly)")
                print(f"Current validation loss: {self.training_variables.val_loss}")
                print(f"Current max length: {self.training_variables.max_len_start}")
                print(f"max length before: {self.train_dataloader.batch_sampler.max_length}")
                self.train_dataloader.batch_sampler.max_length = self.training_variables.max_len_start # Important! this should be refactored for clarity, but for now, I'll update the max len here.
                print(f"max length after: {self.train_dataloader.batch_sampler.max_length}")
                # print(f"loaded optimizer: {self.optimizer}")

        # Now you train the model
        self.transformer.train()
        self.vae.eval()
        
        for epoch in range(starting_epoch, self.config.num_epochs):

            if epoch == starting_epoch:
                dataloader = skipped_dataloader
            else:
                dataloader = self.train_dataloader

            progress_bar = tqdm(total=len(dataloader), disable = True) # not self.accelerator.is_local_main_process)
            progress_bar.set_description(f"Epoch {epoch}")

            for step, batch in enumerate(dataloader):

                # Gradient accumulation
                with self.accelerator.accumulate(self.transformer):
                    
                    input_ids: torch.IntTensor = batch['input_ids']
                    attention_mask: torch.BoolTensor = batch['attention_mask']
                    label: torch.IntTensor = batch['label']
                    # if step == 0:
                    #     print("first batch")
                    #     print(f"{input_ids}")

                    vae_encoded = self.vae.encode(x = input_ids,
                                                  attention_mask = attention_mask,
                    )

                    attention_mask = vae_encoded.attention_masks[-1]
                    latent = vae_encoded.latent_dist.sample() # Mode is deterministic TODO: .sample() or .mode()?
                    latent = latent * self.scaling_factor

                    self.training_variables.n_tokens += attention_mask.sum()

                    # Sample noise to add to the images
                    noise = torch.randn(latent.shape, device=latent.device)
                    if self.config.use_batch_optimal_transport:
                        noise = reorder_noise_for_OT(latent, noise)
                    
                    noise = noise * attention_mask.unsqueeze(1) # Mask the noise
                    bs = latent.shape[0]

                    # Sample a random timestep for each latent
                    timesteps = self.sample_timesteps(bs)

                    # Add noise to the clean images according to the noise magnitude at each timestep
                    noisy_latent = self.noise_scheduler.add_noise(latent, noise, timesteps)

                    # Forward pass
                    output: torch.Tensor = self.transformer(hidden_states = noisy_latent,
                                              attention_mask = attention_mask,
                                              timestep = timesteps,
                                              class_labels = label,
                    ).sample
                    
                    # Loss calculation
                    loss = self.loss(output, latent, noise, attention_mask)
                    loss_back = loss * attention_mask.sum() # https://www.reddit.com/r/MachineLearning/comments/1acbzrx/d_gradient_accumulation_should_not_be_used_with/

                    # Backward pass
                    self.accelerator.backward(loss_back)

                    # Gradient clipping and gradient scaling for gradient accumulation with different length batches
                    if self.accelerator.sync_gradients:
                        self.scale_gradients(self.transformer, self.training_variables.n_tokens)
                        self.training_variables.n_tokens = 0
                        if self.config.gradient_clip_val is not None:
                            self.accelerator.clip_grad_norm_(self.transformer.parameters(), self.config.gradient_clip_val)
                    
                    # Update the model
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                    if self.accelerator.sync_gradients:
                        self.ema.update()

                # Log the progress
                progress_bar.update(1)
                logs = {"train_loss": loss.detach().item(), 
                        "lr": self.lr_scheduler.get_last_lr()[0], 
                        "step": self.training_variables.global_step,
                }
                progress_bar.set_postfix(**logs)
                self.accelerator.log(logs, step=self.training_variables.global_step)
                self.training_variables.global_step += 1

                # Update the max_len
                if self.training_variables.global_step % self.config.max_len_doubling_steps == 0:
                    self.update_max_len()

                # Evaluation and saving the model
                if self.training_variables.global_step == 1 or self.training_variables.global_step % self.config.save_image_model_steps == 0 or self.training_variables.global_step == len(self.train_dataloader) * self.config.num_epochs:
                    break # TODO: remove break
            ### My RAM gets eaten somewhere here
                    # Test of inference using the EMA model
                    self.accelerator.wait_for_everyone()
                    if self.accelerator.is_main_process:
                        pipeline = ProtDiffusionPipeline(transformer=self.accelerator.unwrap_model(self.ema.ema_model), vae=self.vae, scheduler=self.noise_scheduler, tokenizer=self.tokenizer)
                        pipeline.transformer.eval()
                        self.inference_test(pipeline)
            ### My RAM gets eaten somewhere here

                    # Evaluation
                    self.accelerator.wait_for_everyone()
                    ema_loss = self.evaluate(self.ema.ema_model)
                    mse_loss = self.evaluate(self.transformer)
                    logs = {"val_loss_ema": ema_loss, 
                            "val_loss_mse": mse_loss,
                    }
                    self.accelerator.log(logs, step=self.training_variables.global_step)
                    if self.accelerator.is_main_process:
                        self.accelerator.save_state()
                    self.transformer.train() # Make sure the model is in train mode

            # Every save epoch
            torch.cuda.empty_cache()
            if epoch % self.config.save_epochs == 0:

            ### My RAM gets eaten somewhere here
                # Test of inference using the MSE model every epoch
                self.accelerator.wait_for_everyone()
                if self.accelerator.is_main_process:
                    pipeline = ProtDiffusionPipeline(transformer=self.accelerator.unwrap_model(self.ema.ema_model), vae=self.vae, scheduler=self.noise_scheduler, tokenizer=self.tokenizer)
                    pipeline.transformer.eval()
                    self.inference_test(pipeline)
            ### My RAM gets eaten somewhere here

                # Evaluation
                self.accelerator.wait_for_everyone()
                ema_loss = self.evaluate(self.ema.ema_model)
                mse_loss = self.evaluate(self.transformer)
                logs = {"val_loss_ema": ema_loss, 
                        "val_loss_mse": mse_loss,
                }
                self.accelerator.log(logs, step=self.training_variables.global_step)
                if self.accelerator.is_main_process:
                    self.accelerator.save_state()
                self.transformer.train() # Make sure the model is in train mode

        # After training
        self.accelerator.wait_for_everyone()
        self.accelerator.end_training()
        self.save_pretrained()

    def save_pretrained(self, output_dir: Optional[str] = None):

        ce_model = self.accelerator.unwrap_model(self.transformer)
        ema_model = self.accelerator.unwrap_model(self.ema.ema_model)

        if output_dir is None:
            output_dir = os.path.join(self.config.output_dir, "pretrained")
        
        os.makedirs(output_dir, exist_ok=True)
        
        ce_model.save_pretrained(os.path.join(output_dir, "CE"))
        ema_model.save_pretrained(os.path.join(output_dir, "EMA"))
# %%

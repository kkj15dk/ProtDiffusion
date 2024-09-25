# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler
import torch.multiprocessing as mp
from ema_pytorch import EMA

from diffusers.optimization import get_cosine_schedule_with_warmup
from datasets import Dataset
from collections import defaultdict

import os
import numpy as np

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
import json

from New1D.autoencoder_kl_1d import AutoencoderKL1D
from visualization_utils import make_logoplot

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

@dataclass
class TrainingConfig:
    batch_size: int = 64  # the batch size
    mega_batch: int = 1000 # how many batches to use for batchsampling
    num_epochs: int = 1  # the number of epochs to train the model
    gradient_accumulation_steps: int = 2  # the number of steps to accumulate gradients before taking an optimizer step
    learning_rate: float = 1e-4  # the learning rate
    lr_warmup_steps:int  = 1000
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

    gradient_clip_val: float = 5.0  # the value to clip the gradients to
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

def encode(example, sequence_key: str, id_key: str, label_key: str, pad_to_multiple_of: int, tokenizer: PreTrainedTokenizerFast):
    output = tokenizer(example[sequence_key],
                        padding = True,
                        truncation=False, # We need to truncate the sequences later, so we set this to False
                        pad_to_multiple_of = pad_to_multiple_of,
                        return_token_type_ids=False,
                        return_attention_mask=False, # We need to attend to padding tokens, so we set this to False
    )
    output['id'] = example[id_key] or None
    label = example[label_key] or None
    if label is not None:
        if label == 2:
            output['label'] = 1
        elif label == 2759:
            output['label'] = 0
        else:
            output['label'] = None
    output['length'] = len(output['input_ids'])
    return output

def group_data(dataset: Dataset, chunk_size: int = 10000, output_dir: str = "grouped_data") -> Dataset:
    print("Grouping data with low memory requirements")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    dataset_dict = defaultdict(lambda: {'label': None, 'input_ids': [], 'lengths': []})
    chunk_index = 0
    
    # Process the dataset in chunks
    for i in tqdm(range(0, len(dataset), chunk_size), desc="Processing chunks"):
        chunk = dataset[i:i + chunk_size]
        
        for example in chunk:
            id = example['id']
            if dataset_dict[id]['label'] is None:
                dataset_dict[id]['label'] = example['label']
            dataset_dict[id]['input_ids'].append(example['input_ids'])
            dataset_dict[id]['lengths'].append(example['length'])
        
        # Write the chunk to disk
        chunk_file = os.path.join(output_dir, f"chunk_{chunk_index}.json")
        with open(chunk_file, 'w') as f:
            json.dump(dataset_dict, f)
        
        # Clear the dictionary for the next chunk
        dataset_dict.clear()
        chunk_index += 1
    
    print(f"Processed {chunk_index} chunks")
    
    # Combine the intermediate results
    combined_dict = defaultdict(lambda: {'label': None, 'input_ids': [], 'lengths': []})
    
    for chunk_file in tqdm(os.listdir(output_dir), desc="Combining chunks"):
        chunk_path = os.path.join(output_dir, chunk_file)
        with open(chunk_path, 'r') as f:
            chunk_data = json.load(f)
            for id, data in chunk_data.items():
                if combined_dict[id]['label'] is None:
                    combined_dict[id]['label'] = data['label']
                combined_dict[id]['input_ids'].extend(data['input_ids'])
                combined_dict[id]['lengths'].extend(data['lengths'])
    
    print(f"Grouped data into {len(combined_dict)} clusters based on the identifier")
    
    # Create a generator function
    def data_generator():
        for id, data in tqdm(combined_dict.items(), desc="Creating grouped dataset"):
            yield {
                'id': id,
                'label': data['label'],
                'input_ids': data['input_ids'],
                'lengths': data['lengths']
            }
    
    grouped_dataset = Dataset.from_generator(data_generator)
    return grouped_dataset

def prepare_dataloader(config: TrainingConfig,
                        dataset: Dataset,
                        max_len: int,
                        input_ids_key: str = 'input_ids',
                        drop_last: bool = False,
                        num_workers: int = 1,
    ) -> DataLoader:

    sampler = BatchSampler(dataset,
                            config.batch_size,
                            config.mega_batch,
                            max_length=max_len,
                            input_ids_key=input_ids_key,
                            drop_last=drop_last,
                            num_workers=num_workers,
    )
    
    clustered_dataset = ClusteredDataset(dataset, 
                                         input_ids_key=input_ids_key
    )
    
    dataloader = DataLoader(clustered_dataset,
                            batch_sampler=sampler, 
                            collate_fn=sampler.collate_fn,
                            num_workers=num_workers,
    )
    return dataloader

class ClusteredDataset(Dataset):
    '''
    Create a custom dataset for the clustered dataset.
    The dataset is a dictionary with the identifier as the key, and the value is a dictionary with the label, list of sequences, and list of lengths.
    '''
    def __init__(self, dataset, 
                 input_ids_key: str = 'input_ids',
                 label_key: str = 'label',
                 length_key: str = 'length',
    ):
        self.dataset = dataset
        self.input_ids_key = input_ids_key
        self.label_key = label_key
        self.length_key = length_key

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: Union[List[tuple[int, int]], tuple[int,int], List[int], int]): # Way too convoluted, I'm sorry.
        '''
        Get a sample from the dataset. Using two indices, the first index is the cluster index, and the second index is the sample index.
        
        If you choose a single index, or a list of single integers it will return the entire cluster.
        '''

        if isinstance(idx, tuple):
            if isinstance(idx[0], int) and isinstance(idx[1], int):
                idx = [idx]

        if isinstance(idx, int):
            idx = [(idx, i) for i in range(len(self.dataset[idx][self.label_key]))]

        if isinstance(idx, list):
            if isinstance(idx[0], int):
                idx = [(index, i) for index in idx for i in range(len(self.dataset[index][self.label_key]))]
        
        clusterindex, sampleindex = zip(*idx)

        data = self.dataset[clusterindex]
        length_key = self.length_key
        input_ids_key = self.input_ids_key
        label_key = self.label_key

        id = data['id']

        label = []
        length = []
        input_ids = []
        
        for i in range(len(idx)):
            sampleindex_i = sampleindex[i]

            input_ids.append(data[input_ids_key][i][sampleindex_i])
            length.append(data[length_key][i][sampleindex_i])
            label.append(data[label_key][i][sampleindex_i])

        return {'id': id, 'label': label, 'length': length, 'input_ids': input_ids}

class BatchSampler(Sampler): 
    '''
    BatchSampler for variable length sequences, batching by similar lengths, to prevent excessive padding.
    '''
    def __init__(self, 
                 dataset: Dataset, 
                 batch_size: int, 
                 mega_batch_size: int, 
                 max_length: Optional[int] = None,
                 input_ids_key: str = 'input_ids', 
                 drop_last: bool = True,
                 num_workers: int = 1):
        self.batch_size = batch_size
        self.mega_batch_size = mega_batch_size
        self.drop_last = drop_last
        self.max_length = max_length
        self.input_ids_key = input_ids_key
        self.dataset = dataset
        self.num_workers = num_workers

    def collate_fn(self, batch):
        sample_max_len = max(item['length'] for item in batch)
        max_length_cap = self.max_length
        input_ids_key = self.input_ids_key

        if max_length_cap is not None:
            max_len = min(sample_max_len, max_length_cap)
        else:
            max_len = sample_max_len

        batch_size = len(batch)
        input_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
        attention_mask = torch.zeros(batch_size, max_len, dtype=torch.uint8)
        class_labels = torch.zeros(batch_size, dtype=torch.long)
        identifiers = [item['id'] for item in batch]

        for i, item in enumerate(batch):
            seq_len = item['length']
            input_ids_seq = item[input_ids_key]

            if seq_len > max_len:
                index = random.randint(0, seq_len - max_len)
                input_ids[i, :max_len] = torch.tensor(input_ids_seq[index:index+max_len], dtype=torch.long)
                attention_mask[i, :max_len] = 1
            else:
                input_ids[i, :seq_len] = torch.tensor(input_ids_seq, dtype=torch.long)
                attention_mask[i, :seq_len] = 1

            class_labels[i] = item['label']

        return {
            'id': identifiers, 
            'input_ids': input_ids, 
            'attention_mask': attention_mask, 
            'class_label': class_labels
        }
    
    def get_lengths(self, indices, return_dict, process_id):
        return_dict[process_id] = [self.dataset[i]['length'] for i in indices]

    def __iter__(self):
        size = len(self.dataset)
        indices = list(range(size))
        random.shuffle(indices)

        step = self.mega_batch_size * self.batch_size
        for i in range(0, size, step):
            pool_indices = indices[i:i+step]

            # New implementation
            # Use torch.multiprocessing to get lengths
            manager = mp.Manager()
            return_dict = manager.dict()
            processes = []
            chunk_size = len(pool_indices) // self.num_workers
            chunks = [pool_indices[j:j + chunk_size] for j in range(0, len(pool_indices), chunk_size)]

            for process_id, chunk in enumerate(chunks):
                p = mp.Process(target=self.get_lengths, args=(chunk, return_dict, process_id))
                p.start()
                processes.append(p)

            for p in processes:
                p.join()
            
            # Retrieve results in order
            pool_lengths = [length for process_id in sorted(return_dict.keys()) for length in return_dict[process_id]]

            # # old implementation
            # pool_lengths = [self.dataset[i]['length'] for i in pool_indices]

            sample_indices = [random.randint(0, len(lenlist) - 1) for lenlist in pool_lengths]
            pool_lengths = [lenlist[index] for lenlist, index in zip(pool_lengths, sample_indices)] # list of lengths

            # Zip indices with sample indices and sort by length more efficiently
            pool = list(zip(pool_lengths, pool_indices, sample_indices))
            pool.sort(key=lambda x: x[0])

            pool = [(pool_id, sample_id) for _, pool_id, sample_id in pool]

            mega_batch_indices = list(range(0, len(pool), self.batch_size))
            random.shuffle(mega_batch_indices) # shuffle the mega batches, so that the model doesn't see the same order of lengths every time. The small batch will however always be the one with longest lengths

            for j in mega_batch_indices:
                if self.drop_last and j + self.batch_size > len(pool): # drop the last batch if it's too small
                    continue

                batch = pool[j:j+self.batch_size]
                
                yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        else:
            return (len(self.dataset) + self.batch_size - 1) // self.batch_size

def logoplot_callback(pool, name):
    def callback(result):
        print(f"Logoplot for {name} created at {result}")
        pool.close()
        pool.join()
    return callback

@dataclass
class TrainingVariables:
    Epoch: int = 0
    global_step: int = 0
    val_loss: float = float("inf")
    grads: Optional[torch.Tensor] = None

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
                 config: TrainingConfig, 
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
            optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay, momentum=config.SGDmomentum)
        else:
            raise ValueError("Invalid optimizer, choose between `AdamW`, `Adam`, `SGD`, and `Adamax`")

        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=config.lr_warmup_steps,
            num_training_steps=(len(train_dataloader) * config.num_epochs // config.gradient_accumulation_steps),
        )
        # Prepare everything
        # There is no specific order to remember, you just need to unpack the
        # objects in the same order you gave them to the prepare method.
        self.model, self.optimizer, self.train_dataloader, self.test_dataloader, self.val_dataloader, self.lr_scheduler = self.accelerator.prepare(
            model, optimizer, train_dataloader, test_dataloader, val_dataloader, lr_scheduler
        )
        self.accelerator.register_for_checkpointing(self.training_variables)

    def logits_to_token_ids(self, logits: torch.Tensor, cutoff: Optional[float] = None) -> torch.Tensor:
        '''
        Convert a batch of logits to token_ids.
        Returns token_ids
        '''
        if cutoff is None:
            token_ids = logits.argmax(dim=1)
        else:
            token_ids = torch.where(logits.max(dim=1).values > cutoff, 
                                    logits.argmax(dim=1), 
                                    torch.tensor([self.tokenizer.unknown_token_id])
                                    )
        return token_ids

    def update_max_len(self):
        if self.config.max_len_start < self.config.max_len:
            self.config.max_len_start *= 2
            self.config.max_len_start = min(self.config.max_len_start, self.config.max_len) # To not have an int exploding to infinity in the background
            max_len = min(self.config.max_len_start, self.config.max_len)
            print(f"Updating max_len to {max_len}")
            self.train_dataloader.batch_sampler.max_length = max_len

    @torch.no_grad()
    def evaluate(self,
                 model: AutoencoderKL1D,
    ) -> dict:
        model.eval()
        test_dir = os.path.join(self.config.output_dir, "samples")
        os.makedirs(test_dir, exist_ok=True)

        running_loss = 0.0
        num_correct_residues = 0
        total_residues = 0
        name = f"step_{self.training_variables.global_step//1:08d}"

        progress_bar = tqdm(total=len(self.val_dataloader), disable=not self.accelerator.is_local_main_process)
        progress_bar.set_description(f"Evaluating {name}")

        for i, sample in enumerate(self.val_dataloader):

            output = model(sample = sample['input_ids'],
                                attention_mask = sample['attention_mask'],
                                sample_posterior = False, # Should be set to False in inference. This will take the mode of the latent dist
            )

            ce_loss, kl_loss = model.loss_fn(output, sample['input_ids'])
            loss = ce_loss + kl_loss * self.config.kl_weight
            running_loss += loss.item()
            
            if i == 0:
                logoplot_sample = output.sample[0]
                logoplot_sample_id = sample['id'][0]
                print("shape", logoplot_sample.shape)
                probs = F.softmax(logoplot_sample, dim=1).cpu().numpy()
                pool = mp.Pool(processes=1)
                pool.apply_async(make_logoplot, 
                                [probs, logoplot_sample_id, f"{test_dir}/{name}_logoplot_{logoplot_sample_id}.png"], 
                                callback=logoplot_callback(pool, name)
                )

            token_ids_pred = self.logits_to_token_ids(output.sample, cutoff=self.config.cutoff)

            token_ids_correct = ((sample['input_ids'] == token_ids_pred) & (sample['attention_mask'] == 1)).long()
            num_residues = torch.sum(sample['attention_mask'], dim=1).long()

            num_correct_residues += token_ids_correct.sum().item()
            total_residues += num_residues.sum().item()

            # Decode the predicted sequences, and remove zero padding
            seqs_pred = self.tokenizer.batch_decode(token_ids_pred, skip_special_tokens=self.config.skip_special_tokens)
            seqs_lens = torch.sum(sample['attention_mask'], dim=1).long()
            seqs_pred = [seq[:i] for seq, i in zip(seqs_pred, seqs_lens)]

            # Save all samples as a FASTA file
            seq_record_list = [SeqRecord(Seq(seq), id=str(sample['id'][i]), 
                            description=
                            f"classlabel: {sample['class_label'][i].item()} acc: {token_ids_correct[i].sum().item() / num_residues[i].item():.2f}")
                            for i, seq in enumerate(seqs_pred)]
            with open(f"{test_dir}/{name}.fa", "a") as f:
                SeqIO.write(seq_record_list, f, "fasta")
            
            progress_bar.update(1)
        
        acc = num_correct_residues / total_residues
        print(f"{name}, val_loss: {running_loss / len(self.val_dataloader):.4f}, val_accuracy: {acc:.4f}")
        logs = {"val_loss": loss.detach().item(), 
                "val_ce_loss": ce_loss.detach().item(), 
                "val_kl_loss": kl_loss.detach().item(),
                "val_acc": acc,
                }
        return logs
    
    def train_loop(self, from_checkpoint: Optional[Union[str, os.PathLike]] = None):
  
        # start the loop
        if self.accelerator.is_main_process:
            self.ema = EMA(self.model, 
                           beta = self.config.ema_decay, 
                           update_after_step = self.config.ema_update_after,
                           update_every = self.config.ema_update_every
            )
            self.ema.to(self.accelerator.device)
            self.accelerator.register_for_checkpointing(self.ema)

            # Create the output directory
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

            # load the checkpoint if it exists
            if from_checkpoint is None:
                skipped_dataloader = self.train_dataloader
                starting_epoch = 0
                self.training_variables.global_step = 0
                self.training_variables.val_loss = float("inf")
            else:
                self.accelerator.load_state(input_dir=from_checkpoint)
                # Skip the first batches
                starting_epoch = self.training_variables.global_step // len(self.train_dataloader)
                batches_to_skip = self.training_variables.global_step % len(self.train_dataloader)
                skipped_dataloader = self.accelerator.skip_first_batches(self.train_dataloader, batches_to_skip)
                print(f"Loaded checkpoint from {from_checkpoint}")
                print(f"Starting from step {self.training_variables.global_step}")
                print(f"Validation loss: {self.training_variables.val_loss}")

        # Now you train the model
        self.model.train()
        for epoch in range(starting_epoch, self.config.num_epochs):

            if epoch == starting_epoch:
                dataloader = skipped_dataloader
            else:
                dataloader = self.train_dataloader

            progress_bar = tqdm(total=len(dataloader), disable=not self.accelerator.is_local_main_process)
            progress_bar.set_description(f"Epoch {epoch}")

            for step, batch in enumerate(dataloader):
                
                with self.accelerator.accumulate(self.model):
                    input = batch['input_ids']
                    attention_mask = batch['attention_mask']
                    # Predict the noise residual
                    output = self.model(sample = input,
                                    attention_mask = attention_mask,
                                    sample_posterior = True, # Should be set to true in training
                    )
                    
                    ce_loss, kl_loss = self.model.loss_fn(output, input)
                    
                    kl_weight = self.config.kl_weight * min(1.0, self.training_variables.global_step / self.config.kl_warmup_steps)

                    loss = ce_loss + kl_loss * kl_weight
                    self.accelerator.backward(loss)

                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_val)
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                    if self.accelerator.sync_gradients:
                        self.ema.update()

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

                if self.training_variables.global_step % self.config.max_len_doubling_steps == 0:
                    self.update_max_len()

                if self.training_variables.global_step == 1 or self.training_variables.global_step % self.config.save_image_model_steps == 0 or self.training_variables.global_step == len(self.train_dataloader) * self.config.num_epochs - 1:
                    self.accelerator.wait_for_everyone()

                    logs = self.evaluate(self.ema.ema_model)
                    self.accelerator.log(logs, step=self.training_variables.global_step)

                    new_val_loss = logs["val_loss"]

                    if new_val_loss < self.training_variables.val_loss: # Save the model if the validation loss is lower
                        self.training_variables.val_loss = new_val_loss
                        self.accelerator.save_state(
                            output_dir=self.config.output_dir,
                        )
                    self.model.train() # Set model back to train mode
        self.accelerator.end_training()

# %%

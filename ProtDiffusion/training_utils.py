# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LRScheduler

from diffusers.optimization import get_cosine_schedule_with_warmup

import os
import shutil
import pickle
import numpy as np

from dataclasses import dataclass
from typing import Optional, Literal, Union, List

from transformers import PreTrainedTokenizerFast
from datasets import load_dataset, load_from_disk
import random

from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from grokfast import gradfilter_ema
from tqdm.auto import tqdm

from New1D.autoencoder_kl_1d import AutoencoderKL1D

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
    pad_to_multiple_of: int = 16
    max_len: int = 512  # truncation of the input sequence

    class_embeddings_concat = False  # whether to concatenate the class embeddings to the time embeddings

    push_to_hub = False  # Not implemented yet. Whether to upload the saved model to the HF Hub
    hub_model_id = "kkj15dk/protein-VAE"  # the name of the repository to create on the HF Hub
    hub_private_repo = False
    overwrite_output_dir = True  # overwrite the old model when re-running the notebook
    seed: int = 42

    automatic_checkpoint_naming: bool = True  # whether to automatically name the checkpoints
    total_checkpoints_limit: int = 5  # the total limit of checkpoints to save

    cutoff: Optional[float] = None # cutoff for when to predict the token given the logits, and when to assign the unknown token 'X' to this position
    skip_special_tokens = False # whether to skip the special tokens when writing the evaluation sequences
    kl_weight: float = 0.05 # the weight of the KL divergence in the loss function

    weight_decay: float = 0.01 # weight decay for the optimizer
    grokfast: bool = False # whether to use the grokfast algorithm
    grokfast_alpha: float = 0.98 #Momentum hyperparmeter of the EMA.
    grokfast_lamb: float = 2.0 #Amplifying factor hyperparameter of the filter.


def prepare_dataset(config: TrainingConfig,
                    dataset, 
                    dataset_split: str,
                    dataset_dir: str,
                    tokenizer: PreTrainedTokenizerFast,
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

    sampler = BatchSampler(lengths,
                           config.batch_size,
                           config.mega_batch,
                           tokenizer=tokenizer,
                           max_lenght=config.max_len,
                           drop_last=False)
    dataloader = DataLoader(dataset,
                            batch_sampler=sampler, 
                            collate_fn=sampler.collate_fn)

    return dataloader

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

    def logits_to_token_ids(self, logits: torch.Tensor) -> torch.Tensor:
        '''
        Convert a batch of logits to token_ids.
        Returns token_ids
        '''
        if self.config.cutoff is None:
            token_ids = logits.argmax(dim=1)
        else:
            token_ids = torch.where(logits.max(dim=1).values > self.config.cutoff, 
                                    logits.argmax(dim=1), 
                                    torch.tensor([self.tokenizer.unknown_token_id])
                                    )
        return token_ids

    @torch.no_grad()
    def evaluate(self
    ) -> dict:

        test_dir = os.path.join(self.config.output_dir, "samples")
        os.makedirs(test_dir, exist_ok=True)

        running_loss = 0.0
        num_correct_residues = 0
        total_residues = 0
        name = f"step_{self.training_variables.global_step//1000:04d}k"

        progress_bar = tqdm(total=len(self.val_dataloader), disable=not self.accelerator.is_local_main_process)
        progress_bar.set_description(f"Evaluating {name}")

        for i, sample in enumerate(self.val_dataloader):

            output = self.model(sample = sample['input_ids'],
                                attention_mask = sample['attention_mask'],
                                sample_posterior = True, # Should be set to False in inference
            )

            ce_loss, kl_loss = self.model.loss_fn(output, sample['input_ids'])
            loss = ce_loss + kl_loss * self.config.kl_weight
            running_loss += loss.item()

            token_ids_pred = self.logits_to_token_ids(output.sample)

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
            # Create the output directory
            if not os.path.exists(self.config.output_dir):
                os.makedirs(self.config.output_dir, exist_ok=True)
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
                self.training_variables.grads = None # Initialize the grads for the grokfast algorithm
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
                    loss = ce_loss + kl_loss * self.config.kl_weight
                    self.accelerator.backward(loss)

                    if self.config.grokfast:
                        self.training_variables.grads = gradfilter_ema(self.model, grads=self.training_variables.grads, alpha=self.config.grokfast_alpha, lamb=self.config.grokfast_lamb) 

                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()

                progress_bar.update(1)
                logs = {"train_loss": loss.detach().item(), 
                        "train_ce_loss": ce_loss.detach().item(), 
                        "train_kl_loss": kl_loss.detach().item(), 
                        "lr": self.lr_scheduler.get_last_lr()[0], 
                        "step": self.training_variables.global_step,
                }
                progress_bar.set_postfix(**logs)
                self.accelerator.log(logs, step=self.training_variables.global_step)
                self.training_variables.global_step += 1

                if self.training_variables.global_step == 1 or self.training_variables.global_step % self.config.save_image_model_steps == 0 or self.training_variables.global_step == len(self.train_dataloader) * self.config.num_epochs - 1:
                    self.accelerator.wait_for_everyone()
                    self.model.eval() # Set model to eval mode to generate images
                    logs = self.evaluate()
                    self.accelerator.log(logs, step=self.training_variables.global_step)

                    new_val_loss = logs["val_loss"]

                    if new_val_loss < self.training_variables.val_loss: # Save the model if the validation loss is lower
                        self.training_variables.val_loss = new_val_loss
                        self.accelerator.save_state(
                            output_dir=self.config.output_dir,
                        )
                    self.model.train() # Set model back to train mode

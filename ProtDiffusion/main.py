# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import os
from typing import Optional, Literal, Union, List

from transformers import PreTrainedTokenizerFast
from datasets import load_dataset, load_from_disk
import random

from training_utils import TrainingConfig, BatchSampler, prepare_dataset
from torch.utils.data import DataLoader

config = TrainingConfig()


config.dataset_name = "kkj15dk/test_dataset"
# config.dataset_name = "agemagician/uniref50"
# config.dataset_name = "kkj15dk/UniRef50-encoded"
dataset = load_dataset(config.dataset_name) # , download_mode='force_redownload')
dataset = dataset.shuffle(config.seed)

tokenizer = PreTrainedTokenizerFast.from_pretrained("kkj15dk/protein_tokenizer_new")

dataset_train, train_lengths = prepare_dataset(dataset=dataset, dataset_split='train', dataset_dir='dataset_testcase')
dataset_test, test_lengths = prepare_dataset(dataset=dataset, dataset_split='test', dataset_dir='dataset_testcase')
dataset_val, val_lengths = prepare_dataset(dataset=dataset, dataset_split='val', dataset_dir='dataset_testcase')

test_sampler = BatchSampler(test_lengths,
                            config.batch_size,
                            config.mega_batch,
                            tokenizer=tokenizer,
                            max_lenght=config.max_len,
                            drop_last=False)

val_sampler = BatchSampler(val_lengths,
                            config.batch_size,
                            config.mega_batch,
                            tokenizer=tokenizer,
                            max_lenght=config.max_len,
                            drop_last=False)

train_sampler = BatchSampler(train_lengths,
                            config.batch_size,
                            config.mega_batch,
                            tokenizer=tokenizer,
                            max_lenght=config.max_len,
                            drop_last=False)

test_dataloader = DataLoader(dataset_test,
                            batch_sampler=test_sampler, 
                            collate_fn=test_sampler.collate_fn)
val_dataloader = DataLoader(dataset_val, 
                            batch_sampler=val_sampler,
                            collate_fn=val_sampler.collate_fn)
train_dataloader = DataLoader(dataset_train,
                            batch_sampler=train_sampler,
                            collate_fn=train_sampler.collate_fn)

# %%
from New1D.autoencoder_kl_1d import AutoencoderKL1D

model = AutoencoderKL1D(
    num_class_embeds=tokenizer.vocab_size + 1,  # the number of class embeddings
    
    down_block_types=(
        "DownEncoderBlock1D",  # a regular ResNet downsampling block
        "DownEncoderBlock1D",
        "DownEncoderBlock1D",
        "DownEncoderBlock1D",  # a ResNet downsampling block with spatial self-attention
    ),
    up_block_types=(
        "UpDecoderBlock1D",  # a ResNet upsampling block with spatial self-attention
        "UpDecoderBlock1D",
        "UpDecoderBlock1D",
        "UpDecoderBlock1D",  # a regular ResNet upsampling block
    ),
    block_out_channels=(128, 256, 512, 512),  # the number of output channels for each block
    mid_block_type="UNetMidBlock1D",  # the type of the middle block
    mid_block_channels=1024,  # the number of output channels for the middle block
    mid_block_add_attention=False,  # whether to add a spatial self-attention block to the middle block
    layers_per_block=2,  # how many ResNet layers to use per UNet block
    transformer_layers_per_block=1, # how many transformer layers to use per ResNet layer. Not implemented yet.

    latent_channels=128,  # the dimensionality of the latent space

    num_attention_heads=1,  # the number of attention heads in the spatial self-attention blocks
    upsample_type="conv", # the type of upsampling to use, either 'conv' (and nearest neighbor) or 'conv_transpose'
    act_fn="gelu",  # the activation function to use
)

# %%
model.children
model.num_parameters()

# %%
sample_image = next(iter(val_dataloader))
print(sample_image.keys())
print(sample_image['id'][0])
print(sample_image['input_ids'][0])
print(sample_image['attention_mask'][0])
print(sample_image['class_label'][0])

# %%
import torch

class_labels = sample_image['class_label'][0].unsqueeze(0)
attention_mask = sample_image['attention_mask'][0].unsqueeze(0)
print(class_labels.shape)
print(attention_mask.shape)

# %%
input_ids = sample_image['input_ids'].to(model.device)
attention_mask = sample_image['attention_mask'].to(model.device)

print(input_ids.shape)
print(attention_mask.shape)

output = model(sample = input_ids,
                attention_mask = attention_mask,
                sample_posterior = True, # Should be set to true in training
)

ce_loss, kl_loss = model.loss_fn(output, input_ids)
print(ce_loss)
print(kl_loss)

# %%
from diffusers.optimization import get_cosine_schedule_with_warmup

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

# %%
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from tqdm.auto import tqdm
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from torch.optim.lr_scheduler import LRScheduler

from grokfast import gradfilter_ema

from typing import Union

@dataclass
class TrainingVariables:
    global_step: int = 0
    val_loss: float = float("inf")
    grads: Optional[torch.Tensor] = None

    def state_dict(self):
        return self.__dict__
    
    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)
training_variables = TrainingVariables()

class VAETrainer:
    def __init__(self, 
                 model: AutoencoderKL1D, 
                 tokenizer: PreTrainedTokenizerFast, 
                 optimizer: Union[torch.optim.Adam, torch.optim.AdamW, torch.optim.SGD, torch.optim.Adamax],
                 lr_scheduler: LRScheduler, 
                 train_dataloader: DataLoader, 
                 val_dataloader: DataLoader, 
                 config: TrainingConfig, 
                 training_variables: TrainingVariables = training_variables,
                 test_dataloader: DataLoader = None
        ):
        self.tokenizer = tokenizer
        self.config = config
        self.training_variables = training_variables
        self.accelerator_config = ProjectConfiguration(
            project_dir=self.config.output_dir,
            logging_dir=os.path.join(self.config.output_dir, "logs"),
            automatic_checkpoint_naming=self.config.automatic_checkpoint_naming,
            total_limit=self.config.total_limit, # Limit the total number of checkpoints to 1
        )
        self.accelerator = Accelerator(
            project_config=self.accelerator_config,
            mixed_precision=self.config.mixed_precision,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            log_with="tensorboard",
        )
        # Prepare everything
        # There is no specific order to remember, you just need to unpack the
        # objects in the same order you gave them to the prepare method.
        self.model, self.optimizer, self.train_dataloader, self.test_dataloader, self.val_dataloader, self.lr_scheduler = self.accelerator.prepare(
            model, optimizer, train_dataloader, test_dataloader, val_dataloader, lr_scheduler
        )
        self.accelerator.register_for_checkpointing(self.training_variables)

    def logits_to_token_ids(self, logits):
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
    
    def train_loop(self, from_checkpoint: Optional[int] = None):
  
        # start the loop
        if self.accelerator.is_main_process:
            os.makedirs(self.config.output_dir, exist_ok=True)
            os.makedirs(self.accelerator_config.logging_dir, exist_ok=True)
            if self.config.push_to_hub:
                raise NotImplementedError("Pushing to the HF Hub is not implemented yet")
            
            if from_checkpoint is not None:
                input_dir = os.path.join(self.config.output_dir, "checkpoints", f'checkpoint_{from_checkpoint}')
                self.accelerator.load_state(input_dir=input_dir)
                print(f"Loaded checkpoint from {input_dir}")
                print(f"Starting from step {self.training_variables.global_step}")
                print(f"Validation loss: {self.training_variables.val_loss}")
            else:
                self.training_variables.global_step = 0
                self.training_variables.val_loss = float("inf")
                self.training_variables.grads = None # Initialize the grads for the grokfast algorithm

        # Now you train the model
        self.model.train()
        for epoch in range(self.config.num_epochs):
            progress_bar = tqdm(total=len(self.train_dataloader), disable=not self.accelerator.is_local_main_process)
            progress_bar.set_description(f"Epoch {epoch}")

            for step, batch in enumerate(self.train_dataloader):

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
                        "lr": lr_scheduler.get_last_lr()[0], 
                        "step": self.training_variables.global_step,
                }
                progress_bar.set_postfix(**logs)
                self.accelerator.log(logs, step=self.training_variables.global_step)
                self.training_variables.global_step += 1

                if self.training_variables.global_step == 1 or self.training_variables.global_step % self.config.save_image_model_steps == 0 or self.training_variables.global_step == len(self.train_dataloader):
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

Trainer = VAETrainer(model, tokenizer, optimizer, lr_scheduler, train_dataloader, val_dataloader, config, training_variables, test_dataloader)

# %%
from accelerate import notebook_launcher

notebook_launcher(Trainer.train_loop, num_processes=1)


# %%
notebook_launcher(Trainer.train_loop(from_checkpoint=2), num_processes=1)



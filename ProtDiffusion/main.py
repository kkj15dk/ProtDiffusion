# %%
from training_utils import TrainingConfig, make_dataloader, set_seed, VAETrainer, count_parameters
from transformers import PreTrainedTokenizerFast

from datasets import load_from_disk

from New1D.autoencoder_kl_1d import AutoencoderKL1D

import os

config = TrainingConfig(
    num_epochs=20,  # the number of epochs to train for
    batch_size=16,
    mega_batch=1000,
    gradient_accumulation_steps=16,
    learning_rate = 1e-4,
    lr_warmup_steps = 100,
    kl_warmup_steps = 100,
    save_image_model_steps=300,
    output_dir=os.path.join("output","protein-VAE-UniRef50_v11.1"),  # the model name locally and on the HF Hub
    total_checkpoints_limit=5, # the maximum number of checkpoints to keep
    gradient_clip_val=1.0,
    max_len=16384 , # 512 * 2**6
    max_len_start=16384,
    max_len_doubling_steps=100,
    ema_decay=0.9999,
    ema_update_after=100,
    ema_update_every=10,
)
set_seed(config.seed) # Set the random seed for reproducibility

dataset = load_from_disk('/home/kkj/ProtDiffusion/datasets/testcase-UniRef50_sorted_encoded_grouped')
dataset = dataset.shuffle(config.seed)

# %%
tokenizer = PreTrainedTokenizerFast.from_pretrained("/home/kkj/ProtDiffusion/ProtDiffusion/tokenizer/tokenizer_v1.2.json")

# Split the dataset into train and temp sets using the datasets library
train_test_split_ratio = 0.2
train_val_test_split = dataset.train_test_split(test_size=train_test_split_ratio, seed=config.seed)
train_dataset = train_val_test_split['train']
temp_dataset = train_val_test_split['test']

# Split the temp set into validation and test sets using the datasets library
val_test_split_ratio = 0.5
val_test_split = temp_dataset.train_test_split(test_size=val_test_split_ratio, seed=config.seed)
val_dataset = val_test_split['train']
test_dataset = val_test_split['test']

# Check dataset lengths
print(f"Train dataset length: {len(train_dataset)}")
print(f"Validation dataset length: {len(val_dataset)}")
print(f"Test dataset length: {len(test_dataset)}")

# %%
print("num cpu cores:", os.cpu_count())
print("setting num_workers to 16")
num_workers = 16
train_dataloader = make_dataloader(config, train_dataset, 
                                      max_len=config.max_len_start, 
                                      num_workers=num_workers,
)
val_dataloader = make_dataloader(config, val_dataset, 
                                    max_len=config.max_len, 
                                    num_workers=1,
)
test_dataloader = make_dataloader(config, test_dataset,
                                      max_len=config.max_len, 
                                      num_workers=1,
)
print("length of train dataloader: ", len(train_dataloader))
# %%
model = AutoencoderKL1D(
    num_class_embeds=tokenizer.vocab_size,  # the number of class embeddings
    
    down_block_types=(
        "DownEncoderBlock1D",
        "DownEncoderBlock1D",
        "DownEncoderBlock1D",
        "DownEncoderBlock1D",  # a ResNet downsampling block
    ),
    up_block_types=(
        "UpDecoderBlock1D",  # a ResNet upsampling block
        "UpDecoderBlock1D",
        "UpDecoderBlock1D",
        "UpDecoderBlock1D",
    ),
    block_out_channels=(128, 256, 512, 512),  # the number of output channels for each block
    mid_block_type="UNetMidBlock1D",  # the type of the middle block
    mid_block_channels=1024,  # the number of output channels for the middle block
    mid_block_add_attention=False,  # whether to add a spatial self-attention block to the middle block
    layers_per_block=2,  # how many ResNet layers to use per UNet block
    transformer_layers_per_block=1, # how many transformer layers to use per ResNet layer. Not implemented yet.

    latent_channels=64,  # the dimensionality of the latent space

    num_attention_heads=1,  # the number of attention heads in the spatial self-attention blocks
    upsample_type="conv", # the type of upsampling to use, either 'conv' (and nearest neighbor) or 'conv_transpose'
    act_fn="swish",  # the activation function to use
)
count_parameters(model) # Count the parameters of the model and print

Trainer = VAETrainer(model, 
                     tokenizer, 
                     train_dataloader, 
                     val_dataloader, 
                     config, 
                     test_dataloader,
)

# %%
if __name__ == '__main__':
    Trainer.train_loop(from_checkpoint='/home/kkj/ProtDiffusion/output/protein-VAE-UniRef50_v11.0/Epoch_19')
    Trainer.save_pretrained()
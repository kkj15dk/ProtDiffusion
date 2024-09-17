# %%
from training_utils import TrainingConfig, prepare_dataloader, set_seed, VAETrainer
from transformers import PreTrainedTokenizerFast

from datasets import load_from_disk

from New1D.autoencoder_kl_1d import AutoencoderKL1D

import os

config = TrainingConfig(
    num_epochs=1,  # the number of epochs to train for
    batch_size=16,
    save_image_model_steps=1000,
    output_dir=os.path.join("output","protein-VAE-UniRef50"),  # the model name locally and on the HF Hub
    total_checkpoints_limit=5,  # the maximum number of checkpoints to keep
    max_len=512,
)
set_seed(config.seed) # Set the random seed for reproducibility

dataset = load_from_disk('/home/kaspe/ProtDiffusion/datasets/UniRef50_encoded_grouped')
dataset = dataset.shuffle(config.seed)

# %%
tokenizer = PreTrainedTokenizerFast.from_pretrained("kkj15dk/protein_tokenizer")

# Split the dataset into train and temp sets using the datasets library
train_test_split_ratio = 0.0002
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
train_dataloader = prepare_dataloader(config, train_dataset)
val_dataloader = prepare_dataloader(config, val_dataset)
test_dataloader = prepare_dataloader(config, test_dataset)

# %%
def loop_over_dataloader(dataloader, num_samples=1):
    for i, batch in enumerate(dataloader):
        if i >= num_samples:
            break
        print(f"Batch {i}")

import timeit
one_loop = timeit.timeit(lambda: loop_over_dataloader(train_dataloader, 1), number=1)
ten_loop = timeit.timeit(lambda: loop_over_dataloader(train_dataloader, 10), number=1)
print(f"Time for one sample: {one_loop}")
print(f"Time for ten samples: {ten_loop}")

# %%
model = AutoencoderKL1D(
    num_class_embeds=tokenizer.vocab_size + 1,  # the number of class embeddings
    
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

    latent_channels=128,  # the dimensionality of the latent space

    num_attention_heads=1,  # the number of attention heads in the spatial self-attention blocks
    upsample_type="conv", # the type of upsampling to use, either 'conv' (and nearest neighbor) or 'conv_transpose'
    act_fn="swish",  # the activation function to use
)

Trainer = VAETrainer(model, 
                     tokenizer, 
                     train_dataloader, 
                     val_dataloader, 
                     config, 
                     test_dataloader)

# %%
Trainer.train_loop()

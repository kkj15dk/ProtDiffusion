# %%
from ProtDiffusion.training_utils import VAETrainingConfig, make_clustered_dataloader, set_seed, VAETrainer, count_parameters
from transformers import PreTrainedTokenizerFast

from datasets import load_from_disk

from ProtDiffusion.models.autoencoder_kl_1d import AutoencoderKL1D

import os

config = VAETrainingConfig(
    num_epochs=5, # the number of epochs to train for
    batch_size=32, # 24 batch size seems to be the max with 16384 as max_len for 32 GB GPU right now. With batch_size=32, it crashes wit CUDA OOM error, TODO: Should look into memory management optimisation.
    mega_batch=1000,
    gradient_accumulation_steps=16,
    learning_rate=1e-5,
    lr_warmup_steps=1000,
    kl_warmup_steps=2000,
    save_image_model_steps=10000,
    output_dir=os.path.join("output","protein-VAE-UniRef50_v9.1_latent-8"), # the model name locally and on the HF Hub
    total_checkpoints_limit=5, # the maximum number of checkpoints to keep
    gradient_clip_val=1.0,
    max_len=8192, # 512 * 16 ((2**4))
    max_len_start=8192,
    max_len_doubling_steps=10000,
    ema_decay=0.9999,
    ema_update_after=100,
    ema_update_every=10,
)
print("Output dir: ", config.output_dir)
set_seed(config.seed) # Set the random seed for reproducibility

# dataset = load_from_disk('/work3/s204514/UniRef50_grouped')
dataset = load_from_disk('/home/kkj/ProtDiffusion/datasets/UniRef50_grouped')
dataset = dataset.shuffle(config.seed)

# %%
# tokenizer = PreTrainedTokenizerFast.from_pretrained("/zhome/fb/0/155603/ProtDiffusion/ProtDiffusion/tokenizer/tokenizer_v4.1")
tokenizer = PreTrainedTokenizerFast.from_pretrained("/home/kkj/ProtDiffusion/ProtDiffusion/tokenizer/tokenizer_v4.1")

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
print("num cpu cores:", os.cpu_count())
print("setting num_workers to 16")
num_workers = 16

train_dataloader = make_clustered_dataloader(config.batch_size,
                                             config.mega_batch,
                                             train_dataset,
                                             tokenizer=tokenizer,
                                             max_len=config.max_len_start,
                                             num_workers=num_workers,
)
val_dataloader = make_clustered_dataloader(config.batch_size,
                                           config.mega_batch,
                                           val_dataset, 
                                           tokenizer=tokenizer,
                                           max_len=config.max_len, 
                                           num_workers=1,
)
test_dataloader = make_clustered_dataloader(config.batch_size,
                                            config.mega_batch,
                                            test_dataset,
                                            tokenizer=tokenizer,
                                            max_len=config.max_len, 
                                            num_workers=1,
)
print("length of train dataloader: ", len(train_dataloader))
print("length of val dataloader: ", len(val_dataloader))
print("length of test dataloader: ", len(test_dataloader))

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

    latent_channels=8,  # the dimensionality of the latent space

    num_attention_heads=1,  # the number of attention heads in the spatial self-attention blocks
    upsample_type="conv", # the type of upsampling to use, either 'conv' (and nearest neighbor) or 'conv_transpose'
    act_fn="swish",  # the activation function to use
    padding_idx=tokenizer.pad_token_id,  # the padding index
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
    Trainer.train()
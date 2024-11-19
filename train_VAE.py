# %%
from ProtDiffusion.training_utils import VAETrainingConfig, make_clustered_dataloader, set_seed, VAETrainer, count_parameters
from transformers import PreTrainedTokenizerFast

from datasets import load_from_disk, Dataset, DatasetDict

from ProtDiffusion.models.autoencoder_kl_1d import AutoencoderKL1D

import os

config = VAETrainingConfig(
    num_epochs=12, # the number of epochs to train for
    batch_size=64, # 24 batch size seems to be the max with 16384 as max_len for 32 GB GPU right now. With batch_size=32, it crashes wit CUDA OOM error, TODO: Should look into memory management optimisation.
    mega_batch=240,
    pad_to_multiple_of=8,
    gradient_accumulation_steps=32,
    optimizer = "AdamW",
    learning_rate=4e-6,
    lr_warmup_steps=10000,
    lr_schedule='cosine_10x_decay',
    kl_warmup_steps=20000,
    kl_weight=1e-6, # https://www.reddit.com/r/StableDiffusion/comments/1bo8d3k/why_not_use_ae_rather_than_vae_in_the_stable/
    kl_schedule='constant_with_warmup',
    save_image_model_steps=100000,
    output_dir=os.path.join("output","protein-VAE-UniRef50_v24.4_latent-4_conv_transpose"), # the model name locally and on the HF Hub
    total_checkpoints_limit=1, # the maximum number of checkpoints to keep
    gradient_clip_val=1.0, # 5.0,
    max_len=2048, # 512 * 8 ((2**3))
    max_len_start=256,
    max_len_doubling_steps=1500000,
    ema_decay=0.9999,
    ema_update_after=10000,
    ema_update_every=100,
)
print("Output dir: ", config.output_dir)
set_seed(config.seed) # Set the random seed for reproducibility

dataset = load_from_disk('/work3/s204514/UniRef50_grouped')
# dataset = load_from_disk('datasets/UniRef50_grouped')
# dataset = load_from_disk('/home/kkj/ProtDiffusion/datasets/UniRef50_grouped')
# dataset = load_from_disk('/home/kkj/ProtDiffusion/datasets/UniRef50-test-bad?_grouped')
dataset = dataset.shuffle(config.seed)

train_dataset = dataset['train']
train_dataset = train_dataset.shuffle(config.seed + 1) # shuffe again for restart
val_dataset = dataset['valid']
test_dataset = dataset['test']

# %%
# tokenizer = PreTrainedTokenizerFast.from_pretrained("/zhome/fb/0/155603/ProtDiffusion/ProtDiffusion/tokenizer/tokenizer_v4.1")
# tokenizer = PreTrainedTokenizerFast.from_pretrained("/home/kkj/ProtDiffusion/ProtDiffusion/tokenizer/tokenizer_v4.1")
tokenizer = PreTrainedTokenizerFast.from_pretrained("/zhome/fb/0/155603/ProtDiffusion/ProtDiffusion/tokenizer/tokenizer_v4.2")
# tokenizer = PreTrainedTokenizerFast.from_pretrained("/home/kkj/ProtDiffusion/ProtDiffusion/tokenizer/tokenizer_v4.2")

# Check dataset lengths
print(f"Train dataset length: {len(train_dataset)}")
print(f"Validation dataset length: {len(val_dataset)}")
print(f"Test dataset length: {len(test_dataset)}")

# %%
print("num cpu cores:", os.cpu_count())
print("setting num_workers to 8")
num_workers = 8

train_dataloader = make_clustered_dataloader(config.batch_size,
                                             config.mega_batch,
                                             train_dataset,
                                             tokenizer=tokenizer,
                                             max_len=config.max_len_start,
                                             num_workers=num_workers,
                                             pad_to_multiple_of=config.pad_to_multiple_of,
                                             random_padding=True,
)
val_dataloader = make_clustered_dataloader(config.batch_size,
                                           config.mega_batch,
                                           val_dataset, 
                                           tokenizer=tokenizer,
                                           max_len=config.max_len, 
                                           num_workers=1,
                                           pad_to_multiple_of=config.pad_to_multiple_of,
                                           random_padding=True,
)
test_dataloader = make_clustered_dataloader(config.batch_size,
                                            config.mega_batch,
                                            test_dataset,
                                            tokenizer=tokenizer,
                                            max_len=config.max_len, 
                                            num_workers=1,
                                            pad_to_multiple_of=config.pad_to_multiple_of,
                                            random_padding=True,
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
        # "DownEncoderBlock1D",  # a ResNet downsampling block
    ),
    up_block_types=(
        # "UpDecoderBlock1D",  # a ResNet upsampling block
        "UpDecoderBlock1D",
        "UpDecoderBlock1D",
        "UpDecoderBlock1D",
    ),
    block_out_channels=(128, 256, 512), #, 1024),  # the number of output channels for each block
    mid_block_type="UNetMidBlock1D",  # the type of the middle block
    mid_block_channels=1024,  # the number of output channels for the middle block
    mid_block_add_attention=False,  # whether to add a spatial self-attention block to the middle block
    layers_per_block=2,  # how many ResNet layers to use per UNet block
    transformer_layers_per_block=1, # how many transformer layers to use per ResNet layer. Not implemented yet.

    latent_channels=4,  # the dimensionality of the latent space

    num_attention_heads=1,  # the number of attention heads in the spatial self-attention blocks
    upsample_type="conv_transpose", # the type of upsampling to use, either 'conv' (and nearest neighbor) or 'conv_transpose'
    act_fn="swish",  # the activation function to use
    padding_idx=tokenizer.pad_token_id,  # the padding index
    pad_to_multiple_of=config.pad_to_multiple_of,
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
    Trainer.train(from_checkpoint='/zhome/fb/0/155603/output/protein-VAE-UniRef50_v24.2_latent-4_conv_transpose/checkpoints/checkpoint_17')
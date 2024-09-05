# %%
from training_utils import TrainingConfig, prepare_dataset, set_seed, VAETrainer
from transformers import PreTrainedTokenizerFast
from sklearn.model_selection import train_test_split

from datasets import load_dataset

from New1D.autoencoder_kl_1d import AutoencoderKL1D

import os

config = TrainingConfig(
    num_epochs=500,  # the number of epochs to train for
    batch_size=16,
    save_image_model_steps=500,
    output_dir=os.path.join("output","protein-VAE-UniRef50-9-relu-conv_transpose"),  # the model name locally and on the HF Hub
    total_checkpoints_limit=5,  # the maximum number of checkpoints to keep
    max_len=32,
)
set_seed(config.seed) # Set the random seed for reproducibility

config.dataset_name = "kkj15dk/test_dataset"
dataset = load_dataset(config.dataset_name) # , download_mode='force_redownload')
dataset = dataset.shuffle(config.seed)

tokenizer = PreTrainedTokenizerFast.from_pretrained("kkj15dk/protein_tokenizer_new")

# Split the dataset into train and temp sets
train_dataset, temp_dataset = train_test_split(dataset, test_size=0.0001, random_state=config.seed)

# Split the temp set into validation and test sets
val_dataset, test_dataset = train_test_split(temp_dataset, test_size=0.5, random_state=config.seed)


train_dataloader = prepare_dataset(config, dataset=train_dataset, dataset_path='UniRef50/train', tokenizer=tokenizer)
test_dataloader = prepare_dataset(config, dataset=test_dataset, dataset_dir='UniRef50/test', tokenizer=tokenizer)
val_dataloader = prepare_dataset(config, dataset=val_dataset, dataset_dir='UniRef50/val', tokenizer=tokenizer)


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
    upsample_type="conv_transpose", # the type of upsampling to use, either 'conv' (and nearest neighbor) or 'conv_transpose'
    act_fn="relu",  # the activation function to use
)

Trainer = VAETrainer(model, 
                     tokenizer, 
                     train_dataloader, 
                     val_dataloader, 
                     config, 
                     test_dataloader)

# %%
Trainer.train_loop()

# %%
Trainer.train_loop(from_checkpoint="output/protein-VAE-UniRef50-8/checkpoints/checkpoint_4")

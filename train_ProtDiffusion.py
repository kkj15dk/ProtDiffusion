# %%
from ProtDiffusion.training_utils import ProtDiffusionTrainingConfig, make_dataloader, set_seed, ProtDiffusionTrainer, count_parameters
from transformers import PreTrainedTokenizerFast
from diffusers import DDPMScheduler

from datasets import load_from_disk

from ProtDiffusion.models.autoencoder_kl_1d import AutoencoderKL1D
from ProtDiffusion.models.dit_transformer_1d import DiTTransformer1DModel

import os

config = ProtDiffusionTrainingConfig(
    num_epochs=5000,  # the number of epochs to train for
    batch_size=16,
    mega_batch=1000,
    gradient_accumulation_steps=16,
    learning_rate = 1e-5,
    lr_warmup_steps = 100,
    kl_warmup_steps = 100,
    save_image_model_steps=5000,
    output_dir=os.path.join("output","ProtDiffusion-PKSs-test_v1.1"),  # the model name locally and on the HF Hub
    total_checkpoints_limit=5, # the maximum number of checkpoints to keep
    gradient_clip_val=1.0,
    max_len=2048, # 512 * 2**6
    max_len_start=2048,
    max_len_doubling_steps=100,
    ema_decay=0.9999,
    ema_update_after=100,
    ema_update_every=10,
)
set_seed(config.seed) # Set the random seed for reproducibility

# dataset = load_from_disk('/home/kkj/ProtDiffusion/datasets/UniRef50_grouped-test')
dataset = load_from_disk('/home/kkj/ProtDiffusion/datasets/PKSs_grouped')
dataset = dataset.shuffle(config.seed)

# %%
tokenizer = PreTrainedTokenizerFast.from_pretrained("/home/kkj/ProtDiffusion/ProtDiffusion/tokenizer/tokenizer_v4.1")

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
train_dataloader = make_dataloader(config, 
                                   train_dataset,
                                   tokenizer=tokenizer,
                                   max_len=config.max_len_start,
                                   num_workers=num_workers,
)
val_dataloader = make_dataloader(config, 
                                 val_dataset, 
                                 tokenizer=tokenizer,
                                 max_len=config.max_len, 
                                 num_workers=1,
)
test_dataloader = make_dataloader(config,
                                  test_dataset,
                                  tokenizer=tokenizer,
                                  max_len=config.max_len, 
                                  num_workers=1,
)
print("length of train dataloader: ", len(train_dataloader))
print("length of val dataloader: ", len(val_dataloader))
print("length of test dataloader: ", len(test_dataloader))

vae = AutoencoderKL1D.from_pretrained('/home/kkj/ProtDiffusion/output/protein-VAE-UniRef50_v18.1/pretrained/EMA')
tokenizer = PreTrainedTokenizerFast.from_pretrained('/home/kkj/ProtDiffusion/ProtDiffusion/tokenizer/tokenizer_v4.1')

# %%
transformer = DiTTransformer1DModel(
    num_attention_heads = 8,
    attention_head_dim = 72,
    in_channels = vae.config.latent_channels,
    num_layers = 8,
    attention_bias = True,
    activation_fn = "gelu-approximate",
    num_classes = 2,
    upcast_attention = False,
    norm_type = "ada_norm_zero",
    norm_elementwise_affine = False,
    norm_eps = 1e-5,
    pos_embed_type = "sinusoidal", # sinusoidal
    num_positional_embeddings = 1024,
    use_rope_embed = True, # RoPE https://github.com/lucidrains/rotary-embedding-torch
)
count_parameters(transformer) # Count the parameters of the model and print

# %%
noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

Trainer = ProtDiffusionTrainer(transformer=transformer,
                               vae=vae,
                               tokenizer=tokenizer,
                               train_dataloader=train_dataloader,
                               val_dataloader=val_dataloader,
                               config=config,
                               test_dataloader=test_dataloader,
                               noise_scheduler = noise_scheduler, # the scheduler to use for the diffusion
                               eval_seq_len = [64, 64, 256, 256, 1024, 1024, 4096, 4096], # the sequence lengths to evaluate on
                               eval_class_labels = [0,1,0,1,0,1,0,1], # the class labels to evaluate on, should be a list the same length as the eval batch size
                               eval_guidance_scale = 4.0, # the scale of the guidance for the diffusion
                               eval_num_inference_steps = 100, # the number of inference steps for the diffusion
)

# %%
if __name__ == '__main__':
    Trainer.train()
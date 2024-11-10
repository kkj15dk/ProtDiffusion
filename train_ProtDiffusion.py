# %%
from transformers import PreTrainedTokenizerFast
from diffusers import DDPMScheduler
import torch

from datasets import load_from_disk

from ProtDiffusion.training_utils import ProtDiffusionTrainingConfig, make_clustered_dataloader, set_seed, ProtDiffusionTrainer, count_parameters
from ProtDiffusion.models.autoencoder_kl_1d import AutoencoderKL1D
from ProtDiffusion.models.dit_transformer_1d import DiTTransformer1DModel
from ProtDiffusion.schedulers.FlowMatchingEulerScheduler import FlowMatchingEulerScheduler

import os

config = ProtDiffusionTrainingConfig(
    num_epochs=1000, # the number of epochs to train for
    batch_size=32,
    mega_batch=200,
    gradient_accumulation_steps=16,
    learning_rate = 1e-4,
    lr_warmup_steps = 1000, # 100
    lr_schedule = 'constant', # 'constant', 'cosine'
    save_image_model_steps = 500,
    save_every_epoch = True,
    output_dir=os.path.join("output","ProtDiffusion-UniRef50-test_v3.1-diff-logitnorm"),  # the model name locally and on the HF Hub
    total_checkpoints_limit=5, # the maximum number of checkpoints to keep
    gradient_clip_val=1.0,
    max_len=4096, # 512 * 2**6
    max_len_start=64,
    max_len_doubling_steps=100,
    ema_decay=0.99,
    ema_update_after=300, # 100
    ema_update_every=10,
    use_batch_optimal_transport=True, #False
    use_logitnorm_timestep_sampling=True,
)
print("Output dir: ", config.output_dir)
set_seed(config.seed) # Set the random seed for reproducibility
generator = torch.Generator().manual_seed(config.seed)

# dataset = load_from_disk('/home/kkj/ProtDiffusion/datasets/ACP_grouped')
# dataset = load_from_disk('/home/kkj/ProtDiffusion/datasets/PKSs_grouped')
dataset = load_from_disk('/home/kkj/ProtDiffusion/datasets/UniRef50_grouped')
# dataset = load_from_disk('/work3/s204514/ACP_grouped')
# dataset = load_from_disk('/work3/s204514/PKSs_grouped')
# dataset = load_from_disk('/work3/s204514/UniRef50_grouped')
train_dataset = dataset.shuffle(config.seed)

# %%
# Get pretrained models
vae = AutoencoderKL1D.from_pretrained('/home/kkj/ProtDiffusion/output/protein-VAE-UniRef50_v9.3/pretrained/EMA')
tokenizer = PreTrainedTokenizerFast.from_pretrained("/home/kkj/ProtDiffusion/ProtDiffusion/tokenizer/tokenizer_v4.1")
# # Get pretrained models
# tokenizer = PreTrainedTokenizerFast.from_pretrained("/zhome/fb/0/155603/ProtDiffusion/ProtDiffusion/tokenizer/tokenizer_v4.1")
# vae = AutoencoderKL1D.from_pretrained('/work3/s204514/protein-VAE-UniRef50_v9.3/pretrained/EMA')

# Split the dataset into train and temp sets using the datasets library
train_test_split_ratio = 0.01
train_val_test_split = dataset.train_test_split(test_size=train_test_split_ratio, seed=config.seed)
train_dataset = train_val_test_split['train']
# temp_dataset = train_val_test_split['test']
val_dataset = train_val_test_split['test']

# # Split the temp set into validation and test sets using the datasets library
# val_test_split_ratio = 0.5
# val_test_split = temp_dataset.train_test_split(test_size=val_test_split_ratio, seed=config.seed)
# val_dataset = val_test_split['train']
# test_dataset = val_test_split['test']

# Check dataset lengths
print(f"Train dataset length: {len(train_dataset)}")

print("num cpu cores:", os.cpu_count())
print("setting num_workers to 16")
num_workers = 16

train_dataloader = make_clustered_dataloader(config.batch_size,
                                             config.mega_batch,
                                             train_dataset,
                                             tokenizer=tokenizer,
                                             max_len=config.max_len_start,
                                             num_workers=num_workers,
                                             seed=config.seed,
                                             shuffle=True,
)
val_dataloader = make_clustered_dataloader(config.batch_size,
                                           config.mega_batch,
                                           val_dataset, 
                                           tokenizer=tokenizer,
                                           max_len=config.max_len, 
                                           num_workers=1,
                                           seed=config.seed,
                                           shuffle=False,
)

print("length of train dataloader: ", len(train_dataloader))
print("length of val dataloader: ", len(val_dataloader))

# %%
transformer = DiTTransformer1DModel(
    num_attention_heads = 8,
    attention_head_dim = 64,
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
    num_positional_embeddings = 256, # TODO: Should change based on max_len
    use_rope_embed = True, # RoPE https://github.com/lucidrains/rotary-embedding-torch
)
count_parameters(transformer) # Count the parameters of the model and print

# %%
noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
# noise_scheduler = FlowMatchingEulerScheduler(num_inference_steps=100)

Trainer = ProtDiffusionTrainer(transformer=transformer,
                               vae=vae,
                               tokenizer=tokenizer,
                               config=config,
                               train_dataloader=train_dataloader,
                               val_dataloader=val_dataloader,
                               test_dataloader=None,
                               noise_scheduler = noise_scheduler, # the scheduler to use for the diffusion
                               eval_seq_len = [2048, 2048, 1024, 1024], # the sequence lengths to evaluate on
                               eval_class_labels = [0,1,0,1], # the class labels to evaluate on, should be a list the same length as the eval batch size
                               eval_guidance_scale = 4.0, # the scale of the guidance for the diffusion
                               eval_num_inference_steps = 100, # the number of inference steps for the diffusion
)

# %%
if __name__ == '__main__':
    Trainer.train()
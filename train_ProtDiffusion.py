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
import math

config = ProtDiffusionTrainingConfig(
    num_epochs=10, # the number of epochs to train for
    batch_size=64,
    mega_batch=240,
    gradient_accumulation_steps=2,
    learning_rate = 1e-4,
    lr_warmup_steps = 100, # 100
    lr_schedule = 'cosine', # 'constant', 'cosine'
    save_image_model_steps = 100000,
    save_every_epoch = True, # Inference test with EMA model
    output_dir=os.path.join("output","ProtDiffusion-IPR036736_RoPE"),  # the model name locally and on the HF Hub
    total_checkpoints_limit=1, # the maximum number of checkpoints to keep
    gradient_clip_val=1.0,
    pad_to_multiple_of=8,
    max_len=2048, # 512 * 2**6
    max_len_start=64,
    max_len_doubling_steps=2000,
    ema_decay=0.9999,
    ema_update_after=200, # 100
    ema_update_every=1,
    use_batch_optimal_transport=False, #True
    use_logitnorm_timestep_sampling=False, #True
)
print("Output dir: ", config.output_dir)
set_seed(config.seed) # Set the random seed for reproducibility
generator = torch.Generator().manual_seed(config.seed)

# %%
# Get pretrained models
vae = AutoencoderKL1D.from_pretrained('/home/kkj/ProtDiffusion/output/EMA_VAE_v24.12')
# vae = AutoencoderKL1D.from_pretrained('/zhome/fb/0/155603/output/protein-VAE-UniRef50_v24.12_latent-4_conv_transpose/pretrained/EMA')

tokenizer = PreTrainedTokenizerFast.from_pretrained("/home/kkj/ProtDiffusion/ProtDiffusion/tokenizer/tokenizer_v4.2")
# tokenizer = PreTrainedTokenizerFast.from_pretrained("/zhome/fb/0/155603/ProtDiffusion/ProtDiffusion/tokenizer/tokenizer_v4.2")

dataset = load_from_disk('/home/kkj/ProtDiffusion/datasets/IPR036736_90_grouped')
# dataset = load_from_disk('/work3/s204514/IPR036736_90_grouped')
dataset = dataset.shuffle(config.seed)

train_dataset = dataset['train']
val_dataset = dataset['valid']
test_dataset = dataset['test']

# Check dataset lengths
print(f"Train dataset length: {len(train_dataset)}")

print("num cpu cores:", os.cpu_count())
print("setting num_workers to 8")
num_workers = 8

train_dataloader = make_clustered_dataloader(config.batch_size,
                                             config.mega_batch,
                                             train_dataset,
                                             tokenizer=tokenizer,
                                             max_len=config.max_len_start,
                                             pad_to_multiple_of=config.pad_to_multiple_of,
                                             num_workers=num_workers,
                                             seed=config.seed,
                                             shuffle=True,
                                             random_padding=True,
)
val_dataloader = make_clustered_dataloader(config.batch_size,
                                           config.mega_batch,
                                           val_dataset, 
                                           tokenizer=tokenizer,
                                           max_len=config.max_len, 
                                           pad_to_multiple_of=config.pad_to_multiple_of,
                                           num_workers=1,
                                           seed=config.seed,
                                           shuffle=False,
                                           random_padding=True,
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
    activation_fn = "gelu", # gelu-approximate
    num_classes = 2,
    upcast_attention = False,
    norm_type = "ada_norm_zero",
    norm_elementwise_affine = False,
    norm_eps = 1e-5,
    pos_embed_type = None, # "sinusoidal"
    num_positional_embeddings = config.max_len // config.pad_to_multiple_of,
    use_rope_embed = True, # RoPE https://github.com/lucidrains/rotary-embedding-torch
)
count_parameters(transformer) # Count the parameters of the model and print

# %%
noise_scheduler = DDPMScheduler(num_train_timesteps=1000, clip_sample=False)
# noise_scheduler = FlowMatchingEulerScheduler(num_inference_steps=100)

Trainer = ProtDiffusionTrainer(transformer=transformer,
                               vae=vae,
                               tokenizer=tokenizer,
                               config=config,
                               train_dataloader=train_dataloader,
                               val_dataloader=val_dataloader,
                               test_dataloader=None,
                               noise_scheduler = noise_scheduler, # the scheduler to use for the diffusion
                               eval_seq_len = [2048, 2048, 256, 256], # the sequence lengths to evaluate on
                               eval_class_labels = [0,1,0,1], # the class labels to evaluate on, should be a list the same length as the eval batch size
                               eval_guidance_scale = 4.0, # the scale of the guidance for the diffusion
                               eval_num_inference_steps = 100, # the number of inference steps for the diffusion
)

# %%
if __name__ == '__main__':
    Trainer.train()
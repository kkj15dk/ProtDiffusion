# %%
from ProtDiffusion.training_utils import ProtDiffusionTrainingConfig, make_clustered_dataloader, set_seed, ProtDiffusionTrainer, count_parameters, ClusteredDataset
from transformers import PreTrainedTokenizerFast
from diffusers import DDPMScheduler
import torch

from datasets import load_from_disk

from ProtDiffusion.models.autoencoder_kl_1d import AutoencoderKL1D
from ProtDiffusion.models.dit_transformer_1d import DiTTransformer1DModel

import os

config = ProtDiffusionTrainingConfig(
    num_epochs=100, # the number of epochs to train for
    batch_size=16,
    mega_batch=50,
    gradient_accumulation_steps=16,
    learning_rate = 1e-4,
    lr_warmup_steps = 200,
    save_image_model_steps = 320,
    save_every_epoch = True,
    output_dir=os.path.join("output","ProtDiffusion-PKSs-test_v1.3"),  # the model name locally and on the HF Hub
    total_checkpoints_limit=5, # the maximum number of checkpoints to keep
    gradient_clip_val=1.0,
    max_len=4096, # 512 * 2**6
    max_len_start=4096,
    max_len_doubling_steps=100,
    ema_decay=0.9999,
    ema_update_after=100,
    ema_update_every=10,
    use_batch_optimal_transport=True,
)
set_seed(config.seed) # Set the random seed for reproducibility
generator = torch.Generator().manual_seed(config.seed)

dataset = load_from_disk('/home/kkj/ProtDiffusion/datasets/UniRef50-test_grouped')
dataset = ClusteredDataset(dataset)

# Split the dataset into train and temp sets using the datasets library
train_test_split_ratio = 0.1
train_val_test_split = dataset.train_test_split(test_size=train_test_split_ratio, seed=config.seed)
train_dataset = train_val_test_split['train']
val_dataset = train_val_test_split['test']

# Check dataset lengths
print(f"Train dataset length: {len(train_dataset)}")
print(f"Validation dataset length: {len(val_dataset)}")

# Get pretrained models
tokenizer = PreTrainedTokenizerFast.from_pretrained('/home/kkj/ProtDiffusion/ProtDiffusion/tokenizer/tokenizer_v4.1')
vae = AutoencoderKL1D.from_pretrained('/home/kkj/ProtDiffusion/output/protein-VAE-UniRef50_v9.3/pretrained/EMA')

print("num cpu cores:", os.cpu_count())
print("setting num_workers to 16")
num_workers = 16

train_dataloader = make_clustered_dataloader(config.batch_size,
                                             config.mega_batch,
                                             train_dataset,
                                             tokenizer=tokenizer,
                                             max_len=config.max_len_start,
                                             num_workers=num_workers,
                                             generator=generator,
)

print("length of train dataloader: ", len(train_dataloader))

# %%
transformer = DiTTransformer1DModel(
    num_attention_heads = 4,
    attention_head_dim = 36,
    in_channels = vae.config.latent_channels,
    num_layers = 4,
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

Trainer = ProtDiffusionTrainer(transformer=transformer,
                               vae=vae,
                               tokenizer=tokenizer,
                               config=config,
                               train_dataloader=train_dataloader,
                               val_dataset=val_dataset, # pass the dataset for evaluation
                            #    test_dataloader=None,
                               noise_scheduler = noise_scheduler, # the scheduler to use for the diffusion
                               eval_seq_len = [4096, 4096, 1024, 1024], # the sequence lengths to evaluate on
                               eval_class_labels = [0,1,0,1], # the class labels to evaluate on, should be a list the same length as the eval batch size
                               eval_guidance_scale = 4.0, # the scale of the guidance for the diffusion
                               eval_num_inference_steps = 5, # the number of inference steps for the diffusion
)

# %%
if __name__ == '__main__':
    Trainer.train()
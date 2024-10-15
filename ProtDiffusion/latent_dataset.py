# %%
from models.autoencoder_kl_1d import AutoencoderKL1D
from datasets import load_from_disk, Dataset
import random
import torch

dataset_dir = '/home/kkj/ProtDiffusion/datasets/testcase-UniRef50_sorted_encoded'

dataset = load_from_disk(dataset_dir)

# %%
# load the model
model = AutoencoderKL1D.from_pretrained('/home/kkj/ProtDiffusion/output/protein-VAE-UniRef50_v11.1/pretrained/CE/')
model.eval()
model.to(model.device)

# %%
# Encode the dataset into latents

def encode_latents(
    model: AutoencoderKL1D,
    example: dict,
    max_len: int = 16384,
):
    if 'input_ids' not in example:
        raise ValueError("The example does not contain the 'input_ids' key.")

    if len(example['input_ids']) > max_len:
        start = random.randint(0, len(example['input_ids']) - max_len)
        input = torch.tensor(example['input_ids'][start:start+max_len], dtype=torch.long).unsqueeze(0)
    else:
        input = torch.tensor(example['input_ids'], dtype=torch.long).unsqueeze(0)
    print(input.shape)
    output = model.forward(input)

    latents = output.latent_dist.mode()
    ce_loss, kl_loss = model.loss_fn(output, input)
    acc = model.accuracy_fn(output, input)

    return {'latents': latents, 'ce_loss': ce_loss, 'kl_loss': kl_loss, 'acc': acc}

# %%
# Encode the dataset into latents
dataset = dataset.select_columns(['input_ids'])
dataset = dataset.map(lambda x: encode_latents(model, x), batched=False)
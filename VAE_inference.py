from ProtDiffusion.models.autoencoder_kl_1d import AutoencoderKL1D
from ProtDiffusion.training_utils import process_sequence
from ProtDiffusion.models.pipeline_protein import logits_to_token_ids
from ProtDiffusion.visualization_utils import make_logoplot

from transformers import PreTrainedTokenizerFast
from datasets import load_from_disk

from tqdm import tqdm
import numpy as np
import random
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
import torch.multiprocessing as mp

import gc

import os

id_key: str = 'proteinid'
length_key: str = 'length'
label_key: str = 'label'
sequence_key: str = 'sequence'
pad_to_multiple_of: int = 16
max_length: int = 8192
kl_weight: float = 0.1
cutoff: float = 0.0
skip_special_tokens: bool = False

tokenizer = PreTrainedTokenizerFast.from_pretrained("/home/kkj/ProtDiffusion/ProtDiffusion/tokenizer/tokenizer_v4.1")

def collate_fn(batch):
    '''
    Collate function for the DataLoader.
    Takes dictionary with the keys id_key, length_key, label_key, and sequence_key.
    The length key has to be already rounded to the amount the sequence will be padded to.
    This has to be done as this value is also needed in the BatchSampler, so th elength has to be precomputed when making the dataset. See convert_csv_to_dataset.py for an example.
    The value of each key is a list.
    Returns a dictionary with the keys 'id', 'label', and 'sequence'.
    '''

    assert all(item[length_key] % pad_to_multiple_of == 0 for item in batch), "The length_key values of the sequences must be a multiple of the pad_to_multiple_of parameter." #TODO: Could be commented out and made an assertiong on the dataset level.

    length = [item[length_key] for item in batch]
    sample_max_len = max(length)
    max_length_cap = max_length

    if max_length_cap is not None:
        max_len = min(sample_max_len, max_length_cap)
    else:
        max_len = sample_max_len

    id = []
    sequence = []
    label = []

    for i, item in enumerate(batch):
        # id
        id.append(item[id_key])

        # label
        label.append(item[label_key])

        # sequence
        seq = process_sequence(item[sequence_key])
        seq_len = item[length_key]

        if seq_len > max_len:
            index = random.randint(0, seq_len - max_len)
            sequence.append(seq[index:index+max_len])
        else:
            sequence.append(seq)

    tokenized = tokenizer(sequence,
                    padding=True,
                    truncation=False, # We truncate the sequences beforehand
                    return_token_type_ids=False,
                    return_attention_mask=True,
                    return_tensors="pt",
    )
    input_ids = tokenized['input_ids']
    attention_mask = tokenized['attention_mask'].to(dtype=torch.bool) # Attention mask should be bool for scaled_dot_product_attention
    label = torch.tensor(label)

    return {
        'id': id, 
        'label': label,
        'sequence': sequence,
        'length': length,
        'input_ids': input_ids,
        'attention_mask': attention_mask,
    }

dataset = load_from_disk('/home/kkj/ProtDiffusion/datasets/PKSs') # load ungrouped dataset
model = AutoencoderKL1D.from_pretrained('/home/kkj/ProtDiffusion/tempmodels/EMA_vae')

dataloader = DataLoader(dataset, batch_size=16, collate_fn=collate_fn)

model.eval()
test_dir = os.path.join("VAE_temp")
os.makedirs(test_dir, exist_ok=True)

running_loss = 0.0
running_loss_ce = 0.0
running_loss_kl = 0.0
num_correct_residues = 0
total_residues = 0
latent_data = []
name = f"VAE_PKSs_inference"

progress_bar = tqdm(total=len(dataloader), disable=False)
progress_bar.set_description(f"Evaluating {name}")

with torch.no_grad():
    for i, batch in enumerate(dataloader):

        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        length = batch['length']

        output = model(sample = input_ids,
                        attention_mask = attention_mask,
                        sample_posterior = True, # Could be set to False in inference. This will take the mode of the latent dist TODO: should sample_posterior be true or false?
        )

        ce_loss, kl_loss = model.loss_fn(output, input_ids)
        loss = ce_loss + kl_loss * kl_weight
        running_loss_ce += ce_loss.item()
        running_loss_kl += kl_loss.item()
        running_loss += loss.item()
        running_loss_ce += ce_loss.item()
        running_loss_kl += kl_loss.item()

        if i == 0: # save the first sample each evaluation as a logoplot
            logoplot_sample = output.sample[0]
            # remove the padding
            logoplot_sample_len = length[0]
            logoplot_sample = logoplot_sample[:,:logoplot_sample_len]
            logoplot_sample_id = batch['id'][0]
            probs = F.softmax(logoplot_sample, dim=0).cpu().numpy()
            pool = mp.Pool(1)
            pool.apply_async(make_logoplot, 
                            args=(
                                probs, 
                                logoplot_sample_id, 
                                f"{test_dir}/{name}_probs_{logoplot_sample_id}.png", 
                                tokenizer.decode(range(tokenizer.vocab_size)),
                            ),
                            error_callback=lambda e: print(e),
                            callback=lambda _: pool.close(),
            )
            gc.collect()

        token_ids_pred = logits_to_token_ids(output.sample, tokenizer, cutoff=cutoff)

        token_ids_correct = ((input_ids == token_ids_pred) & (attention_mask == 1)).long()
        num_residues = torch.sum(attention_mask, dim=1).long()

        num_correct_residues += token_ids_correct.sum().item()
        total_residues += num_residues.sum().item()

        # get the latent space data
        latent = output.latent_dist.sample() # TODO: sample or mode?
        mask = output.attention_masks[-1].unsqueeze(1).expand_as(latent)
        data = latent[mask].tolist()
        latent_data.extend(data)

        # Decode the predicted sequences, and remove zero padding
        seqs_pred = tokenizer.batch_decode(token_ids_pred, skip_special_tokens=skip_special_tokens)
        seqs_lens = length

        # Remove the padding from the sequences
        seqs_pred = [seq[:i] for seq, i in zip(seqs_pred, seqs_lens)]

        # Save all samples as a FASTA file
        seq_record_list = [SeqRecord(Seq(seq), id=str(batch['id'][i]), 
                        description=
                        f"label: {batch[label_key][i]} acc: {token_ids_correct[i].sum().item() / num_residues[i].item():.2f}")
                        for i, seq in enumerate(seqs_pred)]

        with open(f"{test_dir}/{name}.fa", "a") as f:
            SeqIO.write(seq_record_list, f, "fasta")

        progress_bar.update(1)

# Calculate statistics for the validation set
mu = np.mean(latent_data)
std = np.std(latent_data)

acc = num_correct_residues / total_residues
log_loss = running_loss / len(dataloader)
log_loss_ce = running_loss_ce / len(dataloader)
log_loss_kl = running_loss_kl / len(dataloader)
print(f"{name}, val_loss: {log_loss:.4f}, val_accuracy: {acc:.4f}, val_mu: {mu:.4f}, val_std: {std:.4f}")
logs = {"val_loss": log_loss, 
        "val_ce_loss": log_loss_ce, 
        "val_kl_loss": log_loss_kl,
        "val_acc": acc,
        "val_mu": mu,
        "val_std": std,
}
with open(f"{test_dir}/{name}_logs.txt", "w") as f:
    f.write(str(logs))
print(f"Validation done")
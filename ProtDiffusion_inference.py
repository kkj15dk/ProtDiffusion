from ProtDiffusion.models.autoencoder_kl_1d import AutoencoderKL1D
from ProtDiffusion.models.dit_transformer_1d import DiTTransformer1DModel
from ProtDiffusion.models.pipeline_protein import ProtDiffusionPipeline, logits_to_token_ids
from ProtDiffusion.visualization_utils import make_logoplot

from transformers import PreTrainedTokenizerFast
from diffusers import DDPMScheduler
import torch
import torch.nn.functional as F

from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO

import os

transformer = DiTTransformer1DModel.from_pretrained('/home/kkj/ProtDiffusion/tempmodels/EMA_transformer')
vae = AutoencoderKL1D.from_pretrained('/home/kkj/ProtDiffusion/tempmodels/EMA_vae')
tokenizer = PreTrainedTokenizerFast.from_pretrained("/home/kkj/ProtDiffusion/ProtDiffusion/tokenizer/tokenizer_v4.1")
noise_scheduler = DDPMScheduler(num_train_timesteps=1000)


pipeline = ProtDiffusionPipeline(
    transformer=transformer,
    vae=vae,
    scheduler=noise_scheduler,
    tokenizer=tokenizer,
)
test_dir = os.path.join("temp")
os.makedirs(test_dir, exist_ok=True)

seqs_lens = [2048, 2048]
class_labels = [0,1]
guidance_scale = 4.0
eval_num_inference_steps = 1000
name = f"Test"

output = pipeline(seq_len=seqs_lens,
                  class_labels=class_labels,
                  guidance_scale=guidance_scale,
                  num_inference_steps=eval_num_inference_steps,
                  generator=None,
                  output_type='logits',
)
logits = output.seqs

# make a logoplot of the first sample
logoplot_sample = logits[0]
# remove the padding
logoplot_sample_len = seqs_lens[0]
logoplot_sample = logoplot_sample[:,:logoplot_sample_len]
logoplot_sample_cl = class_labels[0]
probs = F.softmax(logoplot_sample, dim=0).cpu().numpy()
make_logoplot(probs, 
              name, 
              f"{test_dir}/{name}_length_{logoplot_sample_len}_class_label_{logoplot_sample_cl}.png", 
              tokenizer.decode(range(tokenizer.vocab_size)),
)

token_ids_pred = logits_to_token_ids(logits, tokenizer, cutoff=0)

# Decode the predicted sequences, and remove zero padding
seqs_pred = tokenizer.batch_decode(token_ids_pred, skip_special_tokens=False)

# Remove the padding from the sequences
# seqs_pred = [seq[:i] for seq, i in zip(seqs_pred, seqs_lens)]

# Save all samples as a FASTA file
seq_record_list = [SeqRecord(Seq(seq), id=str(seqs_lens[i]), 
                description=
                f"length: {seqs_lens[i]} label: {class_labels[i]}")
                for i, seq in enumerate(seqs_pred)]

with open(f"{test_dir}/{name}.fa", "a") as f:
    SeqIO.write(seq_record_list, f, "fasta")

print(f"Inference done")
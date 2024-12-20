# Datasets
- Download Uniref100, 90, and 50
- Add [BOP] and [EOP] to start and end, unless it is a fragment
- Padding '-' character to length (8 or 64, depending on if i use DiT or U-Net)
- Sort for very long sequences
- Length as additional field

# Tokenizer
- How the fuck do you do random truncation to some subset of the sequence
- Pad to longest sequence in batch, and make attention/loss mask

# Models
- VAE
    - Make it be deterministic in inference (using mode for sampling posterior)
    - Still have it sample in model.eval(), to be able to calculate kl_loss
    - sigma-VAE ??
    - get the scale_factor, and pass to the diffusion model to scale the latents - see section D.1 and G https://arxiv.org/pdf/2112.10752
    - fix the attention_mask mess. There will be a different attention_mask for the input, as there should be for the output, when an unknown amino acid is in the input. Alternatively, remove all the *attention_mask from the VAE.
    - How should a VAE be used in inference? - With .mode() or .sample()?
- U-Net
- DiT
- FlowMatchDiscreteEulerScheduler

# Layers
- SelfAttention1d
    - head_dim
    - num_layers
    - QK RMSNorm
    - RoPE
- AttenXBlock1d
    - num_attention_heads
    - x-transformers ??
- Masked (Group)normalization https://github.com/pytorch/pytorch/issues/81985

# Training
- [x] Be able to restart training 
    - (somewhat implemented, however, training does not start from the same point in the dataloader, and will be different, even if the seed is the same)
- [ ] Easy test-case
- Better visualisation
 - of the model (architecture, tensorboard can do this?)
 - of confusion matrixes

# Long term
- Inpianting
- CLIP model for proteins (taxonomy, GO, InterPro, PFAM, etc.)
- VAE for downscaling proteins, then diffusion, then upscaling (like stable diffusion)
- KAN vs MLP
- GROKFAST
- NF4 format for smaller size
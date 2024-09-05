## ProtDiffusion

A protein latent diffusion transformer, operating on protein sequences.

The model consists of 2 modules and a tokenizer, with the overall architecture being closely related to the Stable Diffusion pipelines.
The first model is a VAE, trained to compress discrete amino acid sequences to a constinuous latent space, with a certain latent dim, and a certain compression factor, with which the length is compressed by.

The second model is a Diffusion Transformer (DiT), trained in this latent space, to be able to generate plausible protein sequences.
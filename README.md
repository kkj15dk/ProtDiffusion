## ProtDiffusion

A protein latent diffusion transformer, operating on protein sequences.

The model consists of 2 modules and a tokenizer, with the overall architecture being closely related to the Stable Diffusion pipelines.
The first model is a VAE, trained to compress discrete amino acid sequences to a constinuous latent space, with a certain latent dim, and a certain compression factor, with which the length is compressed by.

The second model is a Diffusion Transformer (DiT), trained in this latent space, to be able to generate plausible protein sequences. (wip)

First VAE (9.3) did not work well for a Diffusion model. I think the latent space was very hard for the diffusion model to understand. See 	
parlancex's comment here for more information: https://news.ycombinator.com/item?id=39215949

In essence, the VAE needs to do perceptual compression - something which is is good at. For the semantic compression, we want this to be constained to the Diffusion process, whihc is way better at this, and might be even better in a more compressed latent space.

## License

As this work is built on top of the Huggingface Diffusers library (https://github.com/huggingface/diffusers), the part of the project situated in ProtDiffusion/New1D is under the Apache License. This part started as a clone of the diffusers library, which was then altered heavily to accomodate 1 dimensional, discrete protein sequence data. However, the rest of the project is under the MIT License.
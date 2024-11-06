# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from dataclasses import dataclass

from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, BaseOutput
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers import DDPMScheduler


from transformers import PreTrainedTokenizerFast

from .dit_transformer_1d import DiTTransformer1DModel
from .autoencoder_kl_1d import AutoencoderKL1D
from ..schedulers.FlowMatchingEulerScheduler import FlowMatchingEulerScheduler

from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO

import os

@dataclass
class ProteinPipelineOutput(BaseOutput):
    """
    Output class for latents pipelines.

    Args:
        seqs (`torch.Tensor`):
            Tensor of denoised seqs of length `batch_size`.
    """

    seqs: Union[List[str], List[List[int]], List[torch.Tensor]] # aa_seqs, token_ids, or logits

def logits_to_token_ids(logits: torch.Tensor, tokenizer: PreTrainedTokenizerFast, cutoff: Optional[float] = None) -> torch.Tensor:
    '''
    Convert a batch of logits to token_ids.
    Returns token_ids
    '''
    probs = F.softmax(logits, dim=-2)

    if cutoff is None:
        token_ids = probs.argmax(dim=-2)
    else:
        token_ids = torch.where(probs.max(dim=-2).values > cutoff, 
                                probs.argmax(dim=-2), 
                                torch.tensor([tokenizer.unk_token_id]) # TODO: fix with no unknown_token_id in tokenizer
                                )
    return token_ids

class ProtDiffusionPipeline(DiffusionPipeline):
    r"""
    Pipeline for protein sequence generation.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Parameters:
        transformer ([`DITTransformer1DModel`]):
            A `DITTransformer1DModel` to denoise the encoded latents latents.
        vae ([`AutoencoderKL1D`])
            Variational Auto-Encoder (VAE) model to encode and decode input_ids to and from latent representations.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded latents. Can be one of
            [`DDPMScheduler`], or [`DDIMScheduler`].
        tokenizer ([`PreTrainedTokenizer`]):
            The tokenizer to be used for decoding the sequences into input_ids.
    """

    model_cpu_offload_seq = "transformer->vae"

    def __init__(self,
                 transformer: DiTTransformer1DModel, 
                 vae: AutoencoderKL1D,
                 scheduler: Union[DDPMScheduler, FlowMatchingEulerScheduler],
                 tokenizer: PreTrainedTokenizerFast,
    ):
        super().__init__()
        self.register_modules(transformer=transformer, vae=vae, scheduler=scheduler, tokenizer=tokenizer)

    @torch.no_grad()
    def __call__(
        self,
        seq_len: Union[int, List[int]],
        class_labels: Optional[Union[int, List[int]]] = None,
        guidance_scale: float = 4.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        num_inference_steps: int = 1000,
        output_type: Optional[str] = "aa_seq", # "aa_seq", "token_ids", "logits"
        return_dict: bool = True,
        cutoff: Optional[int] = None,
    ) -> Union[ProteinPipelineOutput, Tuple]:
        r"""
        The call function to the pipeline for generation.

        Args:
            seq_len (List[int]):
                The length of the generated latents. Will be padded to be divisible by the pipelines pad_to_multiple_of attribute.
            Class_labels (`List[int]`, *optional*):
                The class labels to condition the generation on. If not provided, the model will generate latents
                without any class labels.
            guidance_scale (`float`, *optional*, defaults to 4.0):
                The scale of the guidance loss. A higher value will make the model more likely to generate latents
                that follow the class_label. A value of 0 will disable guidance.
            generator (`torch.Generator`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            num_inference_steps (`int`, *optional*, defaults to 1000):
                The number of denoising steps. More denoising steps usually lead to a higher quality latents at the
                expense of slower inference.
            output_type (`str`, *optional*, defaults to `"aa_seq"`):
                The output format of the generated latents. Defaults to aa_seq, amino acid sequence.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.ImagePipelineOutput`] instead of a plain tuple.
            cut_off (`int`, *optional*):
                The per residue cut-off value of the generated latents. If provided, every residue with a value
                below the cut-off will be latents to 'tokenizer.unkown_id', otherwise, the residue will be returned with the highest probability.

        # # Example:

        # # ```py
        # # >>> from diffusers import DDPMPipeline

        # # >>> # load model and scheduler
        # # >>> pipe = DDPMPipeline.from_pretrained("google/ddpm-cat-256")

        # # >>> # run pipeline in inference (sample random noise and denoise)
        # # >>> latents = pipe().images[0]

        # # >>> # save latents
        # # >>> latents.save("ddpm_generated_image.png")
        ```

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.ImagePipelineOutput`] is returned, otherwise a `tuple` is
                returned where the first element is a list with the generated images
        """
        # Preprocess inputs
        if isinstance(seq_len, int):
            seq_len = [seq_len]
        batch_size = len(seq_len)
        if isinstance(class_labels, int):
            class_labels = [class_labels] * batch_size
        assert len(class_labels) == batch_size, "You have to give as many class_labels as batch_size"

        if class_labels is None:
            class_labels = [self.transformer.num_classes] * batch_size # default to the last class, which is the calss corresponding to None
        for i in range(len(class_labels)): # Alternatively, use -1 to represent None
            if class_labels[i] == -1:
                class_labels[i] = self.transformer.num_classes

        for i, length in enumerate(seq_len):
            if length % self.vae.pad_to_multiple_of != 0:
                print(f"{seq_len[i]} is not divisible by {self.vae.pad_to_multiple_of}. Padding to the next multiple.")
                seq_len[i] = seq_len[i] + (self.vae.pad_to_multiple_of - seq_len[i] % self.vae.pad_to_multiple_of)

        latent_len = [len // self.vae.pad_to_multiple_of for len in seq_len]
        latent_channels = self.transformer.config.in_channels
        latent_shape = (batch_size, latent_channels, max(latent_len))
        attention_mask = torch.zeros((batch_size, max(latent_len)), device=self._execution_device, dtype=torch.bool)
        for i, length in enumerate(latent_len):
            attention_mask[i, :length] = 1
        class_labels = torch.tensor(class_labels, device=self._execution_device)

        # Sample gaussian noise to begin loop
        if self._execution_device.type == "mps":
            # randn does not work reproducibly on mps
            latents = randn_tensor(latent_shape, generator=generator)
            latents = latents.to(self._execution_device)
        else:
            latents = randn_tensor(latent_shape, generator=generator, device=self._execution_device)

        # If using guidance, double the input for classifier free guidance
        if guidance_scale > 0: # TODO: Maybe should be above 1 instead of above 0, see https://github.com/huggingface/diffusers/blob/v0.30.3/src/diffusers/pipelines/dit/pipeline_dit.py#L159
            latents = torch.cat([latents] * 2, dim=0)
            attention_mask = torch.cat([attention_mask] * 2, dim=0)
            class_null = torch.tensor([self.transformer.num_classes] * batch_size, device=self._execution_device)
            class_labels = torch.cat([class_labels, class_null], dim=0)
        latents = latents * attention_mask.unsqueeze(1)

        # set step values
        self.scheduler.set_timesteps(num_inference_steps)
        for t in self.progress_bar(self.scheduler.timesteps):
            hidden_states = self.scheduler.scale_model_input(latents, t)
            timesteps = torch.tensor([t] * hidden_states.shape[0], device=self._execution_device)
            
            # 1. predict noise model_output
            model_output = self.transformer(hidden_states = hidden_states, 
                                            attention_mask = attention_mask,
                                            timestep = timesteps,
                                            class_labels = class_labels,
            ).sample

            # 2. Perform guidance
            if guidance_scale > 0: # rest is for a learned sigma, if the transformer has an output for sigma (having 2*latent_channels as output)
                eps, rest = model_output[:, :latent_channels], model_output[:, latent_channels:]
                cond_eps, uncond_eps = eps.chunk(2, dim=0)

                half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
                eps = torch.cat([half_eps, half_eps], dim=0)

                model_output = torch.cat([eps, rest], dim=1)

            # 3. Learned sigma - Not in use right now, see TODO.md
            if self.transformer.out_channels // 2 == latent_channels:
                model_output, _ = model_output.chunk(2, dim=1)

            # 4. compute previous latents: x_t -> x_t-1
            latents = self.scheduler.step(model_output, t, latents, generator=generator).prev_sample

        if guidance_scale > 0:
            latents = latents[:batch_size]
            attention_mask = attention_mask[:batch_size]
            class_labels = class_labels[:batch_size]

        # Decode latents
        if self.vae.config.scaling_factor is not None:
            latents = 1 / self.vae.config.scaling_factor * latents

        vae_output = self.vae.decode(latents, attention_mask=attention_mask).sample

        if output_type == "token_ids":
           output = logits_to_token_ids(vae_output, self.tokenizer, cutoff)
           output = [seq[:, :seq_len[i]] for i, seq in enumerate(output)]
        elif output_type == "logits":
            output = vae_output
            # output = [seq[:, :, :seq_len[i]] for i, seq in enumerate(output)] # TODO: make the abstraction better here. I use this in evaluate, and therefor need to not create a list yet.
        elif output_type == "aa_seq":
            token_ids = logits_to_token_ids(vae_output, self.tokenizer, cutoff)
            output = self.tokenizer.batch_decode(token_ids)
            output = [seq[:seq_len[i]] for i, seq in enumerate(output)]
        else:
            raise ValueError(f"output_type {output_type} not recognized. Must be one of 'token_ids', 'aa_seq', or 'logits'.")


        if not return_dict:
            return (output,)

        return ProteinPipelineOutput(seqs=output)
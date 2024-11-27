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

from .dit_transformer_1d import DiTTransformer1DModel, Transformer1DModelOutput
from .autoencoder_kl_1d import AutoencoderKL1D
from ..schedulers.FlowMatchingEulerScheduler import FlowMatchingEulerScheduler
from ..visualization_utils import latent_ax

from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO

import os
import matplotlib.pyplot as plt

@dataclass
class ProteinPipelineOutput(BaseOutput):
    """
    Output class for latents pipelines.

    Args:
        seqs (`torch.Tensor`):
            Tensor of denoised seqs of length `batch_size`.
    """

    seqs: Union[List[str], List[List[int]], List[torch.Tensor]] # aa_seqs, token_ids, or logits
    hidden_latents: Optional[List[torch.Tensor]] = None,
    attention_mask: Optional[torch.Tensor] = None,
    class_labels: Optional[torch.Tensor] = None,

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
        self.transformer: DiTTransformer1DModel
        self.vae: AutoencoderKL1D
        self.scheduler: Union[DDPMScheduler, FlowMatchingEulerScheduler]
        self.tokenizer: PreTrainedTokenizerFast
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
        return_hidden_latents: bool = False, # wheter to return the latents at each timestep
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
            class_labels = [self.transformer.num_classes] * batch_size # default to the last class, which is the class corresponding to None
        for i in range(len(class_labels)): # Alternatively, use -1 to represent None
            if class_labels[i] == -1:
                class_labels[i] = self.transformer.num_classes

        for i, length in enumerate(seq_len):
            if length % self.vae.pad_to_multiple_of != 0:
                print(f"{seq_len[i]} is not divisible by {self.vae.pad_to_multiple_of}. Padding to the next multiple.")
                seq_len[i] = seq_len[i] + (self.vae.pad_to_multiple_of - seq_len[i] % self.vae.pad_to_multiple_of)

        latent_len = [len // self.vae.pad_to_multiple_of for len in seq_len]
        max_latent_len = max(latent_len)
        latent_channels = self.transformer.config.in_channels
        latent_shape = (batch_size, latent_channels, max_latent_len)
        attention_mask = torch.zeros((batch_size, max_latent_len), device=self._execution_device, dtype=torch.bool)

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

        # If return_hidden_states, save the hidden latents at each timestep
        if return_hidden_latents:
            hidden_latents = []
            hidden_latents.append(latents)
        else:
            hidden_latents = None

        # set step values
        self.scheduler.set_timesteps(num_inference_steps)
        for t in self.progress_bar(self.scheduler.timesteps):
            
            
            hidden_states = self.scheduler.scale_model_input(latents, t)
            timesteps = torch.tensor([t] * hidden_states.shape[0], device=self._execution_device)
            
            # 1. predict noise model_output
            model_output: torch.Tensor = self.transformer(hidden_states = hidden_states, 
                                                          attention_mask = attention_mask,
                                                          timestep = timesteps,
                                                          class_labels = class_labels,
            ).sample

            # 2. Perform guidance
            if guidance_scale > 0:
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

        # If return_hidden_states, save the hidden latents for the finished inference
        if return_hidden_latents:
            hidden_latents.append(latents)

        if guidance_scale > 0:
            latents = latents[:batch_size]
            attention_mask = attention_mask[:batch_size]
            class_labels = class_labels[:batch_size]

        # Decode latents
        if self.vae.config.scaling_factor is not None:
            latents = 1 / self.vae.config.scaling_factor * latents
            if return_hidden_latents:
                hidden_latents = [1 / self.vae.config.scaling_factor * latents for latents in hidden_latents]

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

        return ProteinPipelineOutput(seqs=output, hidden_latents=hidden_latents, attention_mask=attention_mask, class_labels=class_labels)
    
    @torch.no_grad()
    def animate_inference(
        self,
        pipeline_output: ProteinPipelineOutput,
        png_dir: str,
    ):
        assert pipeline_output.hidden_latents is not None, "pipeline_output.hidden_latents must be set to use animate_inference"
        assert pipeline_output.hidden_latents[0].ndim == 3, "pipeline_output.hidden_latents must be a list of 3D tensors"
        assert pipeline_output.hidden_latents[0].shape[1] % 2 == 0, "pipeline_output.hidden_latents must have an even number of channels in the second dimension"

        if not os.path.exists(png_dir):
            os.makedirs(png_dir)
        else:
            print(f"Directory already exists: {png_dir}")
            return

        hidden_batch_size = pipeline_output.hidden_latents[0].shape[0]
        mask_batch_size = pipeline_output.attention_mask.shape[0]
        assert mask_batch_size == 1, "animate_inference only supports batch_size 1, will only animate the first sample in the batch."
        
        assert hidden_batch_size == 2 * mask_batch_size, "hidden_batch_size must be twice mask_batch_size. This indicates that guidance was used, which is required for the animation."
        

        # Get the inputs
        positions_pr_line = 64
        latents = pipeline_output.hidden_latents
        latent_dim = latents[0].shape[1]
        class_labels = pipeline_output.class_labels
        latent_len = latents[0].shape[2]
        pad_to_multiple_of = self.vae.pad_to_multiple_of
        num_lines = (latent_len * pad_to_multiple_of) // positions_pr_line

        n_latent_plots = latent_dim // 2 + 1

        for t in range(len(latents)):

            latent = latents[t]
            guided_latent = latent[1:]
            latent = latent[:1]

            plt.figure(figsize=(100, 5 * num_lines))

            for line in range(num_lines):
                start = line * positions_pr_line
                end = min(start + positions_pr_line, num_positions)
                
                df = pd.DataFrame(array.T[start:end], columns=amino_acids, dtype=float)
                
                logo = logomaker.Logo(df, 
                                    ax=axes[line, 0],
                                    color_scheme=make_color_dict(cs=characters),
                )
                
                logo.style_spines(visible=False)
                logo.style_spines(spines=['left', 'bottom'], visible=True)
                logo.ax.set_ylabel("Probability")
                logo.ax.set_xlabel("Position")
                logo.ax.set_ylim(*ylim)


@torch.no_grad()
def make_logoplot(array, label:str, png_path:str, characters:str = "-[]ACDEFGHIKLMNPQRSTVWY", positions_pr_line:int = 64, width:int = 100, ylim:tuple = (-0.1,1.1), dpi:int = 50):
    assert array.ndim == 2

    amino_acids = list(characters)

    if os.path.exists(png_path): # If the file already exists, skip making the logoplot
        print(f"File already exists: {png_path}")
        return

    num_positions = array.shape[1]
    num_lines = (num_positions + positions_per_line - 1) // positions_per_line

    fig, axes = plt.subplots(num_lines, 1, figsize=(width, 5 * num_lines), squeeze=False)

    for line in range(num_lines):
        start = line * positions_per_line
        end = min(start + positions_per_line, num_positions)
        
        df = pd.DataFrame(array.T[start:end], columns=amino_acids, dtype=float)
        
        logo = logomaker.Logo(df, 
                              ax=axes[line, 0],
                              color_scheme=make_color_dict(cs=characters),
        )
        
        logo.style_spines(visible=False)
        logo.style_spines(spines=['left', 'bottom'], visible=True)
        logo.ax.set_ylabel("Probability")
        logo.ax.set_xlabel("Position")
        logo.ax.set_ylim(*ylim)

    plt.tight_layout()
    plt.title(f"{label}")

    # Save the figure as a PNG file
    plt.savefig(png_path, dpi = dpi)
    plt.close(fig)

    gc.collect()  # Force garbage collection

    return
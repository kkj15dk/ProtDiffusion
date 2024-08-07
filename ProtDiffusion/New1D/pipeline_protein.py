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

from dataclasses import dataclass

from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, BaseOutput

@dataclass
class ProteinPipelineOutput(BaseOutput):
    """
    Output class for seq pipelines.

    Args:
        seqs (`torch.Tensor`):
            Tensor of denoised seqs of length `batch_size`.
    """

    seqs: torch.Tensor

class DDPMProteinPipeline(DiffusionPipeline):
    r"""
    Pipeline for seq generation.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Parameters:
        unet ([`UNet2DModel`]):
            A `UNet2DModel` to denoise the encoded seq latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded seq. Can be one of
            [`DDPMScheduler`], or [`DDIMScheduler`].
        tokenizer ([`PreTrainedTokenizer`]):
            The tokenizer to be used for decoding the seqs.
    """

    model_cpu_offload_seq = "unet"

    def __init__(self, unet, scheduler, tokenizer):
        super().__init__()
        self.register_modules(unet=unet, scheduler=scheduler, tokenizer=tokenizer)

    def tensor_to_seq(self, tensor, cutoff = None):
        '''
        Convert a tensor to a seq using the tokenizer.
        '''
        
        if cutoff is None:
            token_ids = tensor.argmax(dim=1)
        else:
            token_ids = torch.where(tensor.max(dim=1).values > cutoff, 
                                    tensor.argmax(dim=1), 
                                    torch.tensor([self.tokenizer.unknown_token_id])
                                    )

        return self.tokenizer.batch_decode(token_ids)

    @torch.no_grad()
    def __call__(
        self,
        batch_size: int = 1,
        seq_len: int = 256,
        class_labels: Optional[torch.Tensor] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        num_inference_steps: int = 1000,
        output_type: Optional[str] = "aa_seq",
        return_dict: bool = True,
        cutoff: Optional[int] = None,
    ) -> Union[ProteinPipelineOutput, Tuple]:
        r"""
        The call function to the pipeline for generation.

        Args:
            batch_size (`int`, *optional*, defaults to 1):
                The number of images to generate.
            seq_len (`int`, *optional*, defaults to 256):
                The length of the generated seq. Must be divisible by the pipelines pad_to_multiple_of attribute.
            generator (`torch.Generator`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            num_inference_steps (`int`, *optional*, defaults to 1000):
                The number of denoising steps. More denoising steps usually lead to a higher quality seq at the
                expense of slower inference.
            output_type (`str`, *optional*, defaults to `"aa_seq"`):
                The output format of the generated seq. Defaults to aa_seq, amino acid sequence.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.ImagePipelineOutput`] instead of a plain tuple.
            cut_off (`int`, *optional*):
                The per residue cut-off value of the generated seq. If provided, every residue with a value
                below the cut-off will be seq to 'X', otherwise, the residue will be returned with the highest probability.

        Example:

        ```py
        >>> from diffusers import DDPMPipeline

        >>> # load model and scheduler
        >>> pipe = DDPMPipeline.from_pretrained("google/ddpm-cat-256")

        >>> # run pipeline in inference (sample random noise and denoise)
        >>> seq = pipe().images[0]

        >>> # save seq
        >>> seq.save("ddpm_generated_image.png")
        ```

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.ImagePipelineOutput`] is returned, otherwise a `tuple` is
                returned where the first element is a list with the generated images
        """
        # Sample gaussian noise to begin loop
        seq_shape = (batch_size, self.unet.config.in_channels, seq_len)

        if self.device.type == "mps":
            # randn does not work reproducibly on mps
            seq = randn_tensor(seq_shape, generator=generator)
            seq = seq.to(self.device)
        else:
            seq = randn_tensor(seq_shape, generator=generator, device=self.device)

        # set step values
        self.scheduler.set_timesteps(num_inference_steps)
        
        attention_mask = torch.ones((batch_size, seq_len), device=self.device)

        for t in self.progress_bar(self.scheduler.timesteps):
            # 1. predict noise model_output
            model_output = self.unet(sample = seq, 
                                    timestep = t,
                                    class_labels = class_labels,
                                    attention_mask = attention_mask,
                                    ).sample

            # 2. compute previous seq: x_t -> x_t-1
            seq = self.scheduler.step(model_output, t, seq, generator=generator).prev_sample

        seq = (seq / 2 + 0.5).clamp(0, 1).cpu()
        if output_type == "aa_seq":
            seq = self.tensor_to_seq(seq, cutoff)
        elif output_type == "tensor":
            pass

        if not return_dict:
            return (seq,)

        return ProteinPipelineOutput(seqs=seq)

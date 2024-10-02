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
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders.single_file_model import FromOriginalModelMixin
from diffusers.utils.accelerate_utils import apply_forward_hook
from diffusers.models.attention_processor import (
    ADDED_KV_ATTENTION_PROCESSORS,
    CROSS_ATTENTION_PROCESSORS,
    Attention,
    AttentionProcessor,
    AttnAddedKVProcessor,
    AttnProcessor,
)
from diffusers.models.modeling_utils import ModelMixin
from .vae import Decoder1D, DecoderOutput, DiagonalGaussianDistribution1D, Encoder1D, AutoencoderKLOutput1D, EncoderKLOutput1D


class AutoencoderKL1D(ModelMixin, ConfigMixin, FromOriginalModelMixin):
    r"""
    A VAE model with KL loss for encoding images into latents and decoding latent representations into images.

    This model inherits from [`ModelMixin`]. Check the superclass documentation for it's generic methods implemented
    for all models (such as downloading or saving).

    Parameters:
        num_class_embeddings (int, *optional*, defaults to 3): Number of class embeddings in the input.
        down_block_types (`Tuple[str]`, *optional*, defaults to `("DownEncoderBlock2D",)`):
            Tuple of downsample block types.
        up_block_types (`Tuple[str]`, *optional*, defaults to `("UpDecoderBlock2D",)`):
            Tuple of upsample block types.
        block_out_channels (`Tuple[int]`, *optional*, defaults to `(64,)`):
            Tuple of block output channels.
        act_fn (`str`, *optional*, defaults to `"silu"`): The activation function to use.
        latent_channels (`int`, *optional*, defaults to 4): Number of channels in the latent space.
    """

    _supports_gradient_checkpointing = True
    _no_split_modules = ["BasicTransformerBlock", "ResnetBlock2D"]

    @register_to_config
    def __init__(
        self,
        num_class_embeds: int = None,  # the number of class embeddings
        down_block_types: Tuple[str] = ("DownEncoderBlock1D",),
        up_block_types: Tuple[str] = ("UpDecoderBlock1D",),
        mid_block_type: str = "UNetMidBlock1D",
        mid_block_add_attention: bool = False,
        block_out_channels: Tuple[int] = (64,),
        mid_block_channels: Optional[int] = None,
        layers_per_block: int = 1,
        transformer_layers_per_block: int = 1,
        act_fn: str = "silu",
        latent_channels: int = 4,
        norm_num_groups: int = 32,
        num_attention_heads: int = 1,
        upsample_type: str = "conv",
        # sample_size: int = 32,
        # scaling_factor: float = 0.18215,
        # shift_factor: Optional[float] = None,
        # latents_mean: Optional[Tuple[float]] = None,
        # latents_std: Optional[Tuple[float]] = None,
        # force_upcast: float = True,
        use_quant_conv: bool = True,
        use_post_quant_conv: bool = True,
    ):
        super().__init__()

        # pass init params to Encoder
        self.encoder = Encoder1D(
            in_channels=block_out_channels[0],
            out_channels=latent_channels,
            down_block_types=down_block_types,
            block_out_channels=block_out_channels,
            mid_block_type=mid_block_type,
            mid_block_channels=mid_block_channels,
            mid_block_add_attention=mid_block_add_attention,
            layers_per_block=layers_per_block,
            transformer_layers_per_block=transformer_layers_per_block,
            norm_num_groups=norm_num_groups,
            act_fn=act_fn,
            double_z=True,
            num_attention_heads=num_attention_heads,
        )

        # pass init params to Decoder
        self.decoder = Decoder1D(
            in_channels=latent_channels,
            out_channels=block_out_channels[0],
            up_block_types=up_block_types,
            block_out_channels=block_out_channels,
            mid_block_type=mid_block_type,
            mid_block_channels=mid_block_channels,
            mid_block_add_attention=mid_block_add_attention,
            layers_per_block=layers_per_block,
            transformer_layers_per_block=transformer_layers_per_block,
            norm_num_groups=norm_num_groups,
            act_fn=act_fn,
            num_attention_heads=num_attention_heads,
            upsample_type=upsample_type,
        )

        self.quant_conv = nn.Conv1d(2 * latent_channels, 2 * latent_channels, 1) if use_quant_conv else None
        self.post_quant_conv = nn.Conv1d(latent_channels, latent_channels, 1) if use_post_quant_conv else None

        self.embedding_in = nn.Embedding(num_class_embeds, block_out_channels[0])
        self.conv_out = nn.Conv1d(block_out_channels[0], num_class_embeds, 1)


    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, (Encoder1D, Decoder1D)):
            module.gradient_checkpointing = value

    @apply_forward_hook
    def encode(
        self, x: torch.Tensor, attention_mask: torch.Tensor = None
    ) -> Union[EncoderKLOutput1D, Tuple[DiagonalGaussianDistribution1D]]:
        """
        Encode a batch of sequences into latents.

        Args:
            x (`torch.Tensor`): Input batch of input_ids.

        Returns:
                The latent representations of the encoded input_ids. A [`~vae.EncoderKLOutput1D`] is returned.
        """
        x = self.embedding_in(x)
        x = x.permute(0, 2, 1) # (batch_size, num_channels, seq_len)
        x = x * attention_mask.unsqueeze(1) # (batch_size, num_channels, seq_len) * (batch_size, 1, seq_len) to set padding to 0 vectors

        h, attention_masks = self.encoder(x, attention_mask)

        if self.quant_conv is not None:
            moments = self.quant_conv(h)
        else:
            moments = h

        posterior = DiagonalGaussianDistribution1D(moments)

        return EncoderKLOutput1D(latent_dist=posterior, attention_masks=attention_masks)

    def _decode(self, z: torch.Tensor, attention_masks: torch.Tensor = None, return_dict: bool = True) -> Union[DecoderOutput, torch.Tensor]:

        if self.post_quant_conv is not None:
            z = self.post_quant_conv(z)

        dec = self.decoder(z, attention_masks)

        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)

    @apply_forward_hook
    def decode(
        self, z: torch.FloatTensor, attention_masks: torch.Tensor = None, return_dict: bool = True, generator=None
    ) -> Union[DecoderOutput, torch.FloatTensor]:
        """
        Decode a batch of images.

        Args:
            z (`torch.Tensor`): Input batch of latent vectors.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a [`~models.vae.DecoderOutput`] instead of a plain tuple.

        Returns:
            [`~models.vae.DecoderOutput`] or `tuple`:
                If return_dict is True, a [`~models.vae.DecoderOutput`] is returned, otherwise a plain `tuple` is
                returned.

        """
        decoded = self._decode(z, attention_masks).sample

        if not return_dict:
            return (decoded,)

        return DecoderOutput(sample=decoded)

    def loss_fn(
        self,
        output: AutoencoderKLOutput1D, 
        input_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        ce_loss = nn.functional.cross_entropy(output.sample, input_ids, reduction='none') # B, L
        ce_loss = ce_loss * output.attention_masks[0] # B, L
        ce_loss = torch.sum(ce_loss) / output.attention_masks[0].sum()
        
        kl_loss = output.latent_dist.kl()
        kl_loss = kl_loss * output.attention_masks[-1] # B, L
        kl_loss = torch.sum(kl_loss) / output.attention_masks[-1].sum()

        return ce_loss, kl_loss

    def forward(
        self,
        sample: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        sample_posterior: bool = False, # Should be False for inference
        return_dict: bool = True,
        generator: Optional[torch.Generator] = None,
    ) -> Union[AutoencoderKLOutput1D, torch.Tensor]:
        r"""
        Args:
            sample (`torch.Tensor`): Input sample.
            attention_mask (`torch.Tensor`, *optional*):
                Attention mask for the input sample.
            sample_posterior (`bool`, *optional*, defaults to `False`):
                Whether to sample from the posterior.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`DecoderOutput`] instead of a plain tuple.
        """
        x = sample

        output = self.encode(x, attention_mask)
        posterior = output.latent_dist
        attention_masks = output.attention_masks

        if self.training or sample_posterior:
            z = posterior.sample(generator=generator)
        else:
            z = posterior.mode()
        dec = self.decode(z, attention_masks).sample
        logits = self.conv_out(dec)

        if not return_dict:
            return (logits,)
        
        return AutoencoderKLOutput1D(sample=logits, latent_dist=posterior, attention_masks=attention_masks)
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
    A VAE model with KL loss for encoding sequences into latents and decoding latent representations into sequences.

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
        num_class_embeds: int,  # the number of class embeddings
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
        scaling_factor: Optional[float] = None, # 0.18215, TODO: maybe implement scaling factor from the VAE latent sigma
        # shift_factor: Optional[float] = None,
        # latents_mean: Optional[Tuple[float]] = None,
        # latents_std: Optional[Tuple[float]] = None,
        # force_upcast: float = True,
        use_quant_conv: bool = True,
        use_post_quant_conv: bool = True,
        padding_idx: int = 0, # padding index for the input, used to make the embedding of the padding index 0
        pad_to_multiple_of: int = 16,
    ):
        super().__init__()
        self.pad_to_multiple_of = pad_to_multiple_of
        self.padding_idx = padding_idx

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

        self.embedding_in = nn.Embedding(num_class_embeds, block_out_channels[0], padding_idx=self.padding_idx)
        self.conv_out = nn.Conv1d(block_out_channels[0], num_class_embeds, 1)


    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, (Encoder1D, Decoder1D)):
            module.gradient_checkpointing = value

    @apply_forward_hook
    def encode(
        self, x: torch.Tensor, attention_mask: torch.Tensor = None
    ) -> EncoderKLOutput1D:
        """
        Encode a batch of sequences into latents.

        Args:
            x (`torch.Tensor`): Input batch of input_ids.

        Returns:
                The latent representations of the encoded input_ids. A [`~vae.EncoderKLOutput1D`] is returned.
        """
        x = self.embedding_in(x)
        x = x.permute(0, 2, 1) # (batch_size, num_channels, seq_len)
        if attention_mask is not None:
            x = x * attention_mask.unsqueeze(1) # (batch_size, num_channels, seq_len) * (batch_size, 1, seq_len) to set pad_token and unk_token to 0 vectors

        h, attention_masks = self.encoder(x, attention_mask)

        if self.quant_conv is not None:
            moments = self.quant_conv(h)
        else:
            moments = h

        if attention_masks[-1] is not None:
            moments = moments * attention_masks[-1].unsqueeze(1) # (batch_size, 2*num_channels, seq_len) * (batch_size, 1, seq_len) to set padding to 0 vectors

        posterior = DiagonalGaussianDistribution1D(moments)

        return EncoderKLOutput1D(latent_dist=posterior, attention_masks=attention_masks)

    def _decode(self, z: torch.Tensor, attention_mask: torch.Tensor = None, return_dict: bool = True) -> Union[DecoderOutput, torch.Tensor]:
        
        if attention_mask is not None:
            z = z * attention_mask.unsqueeze(1) # TODO: second added 13/10

        if self.post_quant_conv is not None:
            z = self.post_quant_conv(z)
            # if attention_mask is not None:
                # z = z * attention_mask.unsqueeze(1)

        dec = self.decoder(z, attention_mask)

        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)

    @apply_forward_hook
    def decode(
        self, z: torch.FloatTensor, attention_mask: torch.Tensor = None, return_dict: bool = True, generator=None
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
        if attention_mask is not None: # Necessary for inference to set padding to 0 vectors when different seq_len in a batch
            z = z * attention_mask.unsqueeze(1)
        sample = self._decode(z, attention_mask).sample
        decoded = self.conv_out(sample)

        if not return_dict:
            return (decoded,)

        return DecoderOutput(sample=decoded)

    def loss_fn(
        self,
        output: AutoencoderKLOutput1D, 
        input_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        ce_loss = nn.functional.cross_entropy(output.sample, input_ids, reduction='none', ignore_index=self.padding_idx) # B, L
        if output.attention_masks[0] is not None:
            ce_loss = ce_loss * output.attention_masks[0] # B, L
            ce_loss = torch.sum(ce_loss) / output.attention_masks[0].sum()
        else:
            ce_loss = ce_loss.mean()

        kl_loss = output.latent_dist.kl() # B, L
        if output.attention_masks[-1] is not None:
            kl_loss = kl_loss * output.attention_masks[-1] # B, L
            kl_loss = torch.sum(kl_loss) / output.attention_masks[-1].sum()
            # kl_loss = torch.sum(kl_loss, dim=-1) # B : sum along the sequence length, for the correct implementation of the KL loss
            # kl_loss = kl_loss.mean() # take the mean over the batch
        else:
            kl_loss = torch.sum(kl_loss) / output.attention_masks[-1].sum()
            # kl_loss = torch.sum(kl_loss, dim=-1) # B : sum along the sequence length, for the correct implementation of the KL loss
            # kl_loss = kl_loss.mean()

        return ce_loss, kl_loss
    
    def accuracy_fn(
        self,
        output: AutoencoderKLOutput1D,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        mask = output.attention_masks[0]
        logits = output.sample
        pred = torch.argmax(logits, dim=1)
        correct = (pred == input_ids).float()
        if mask is not None:
            correct = correct * mask
            return correct.sum() / mask.sum()
        else:
            return correct.mean()

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
        assert sample.shape[-1] % self.pad_to_multiple_of == 0, f"Input seq_len must be divisible by {self.pad_to_multiple_of}"
        x = sample

        output = self.encode(x, attention_mask)
        posterior = output.latent_dist
        attention_masks = output.attention_masks

        if self.training or sample_posterior: # Overriding sample_posterior to True for training
            z = posterior.sample(generator=generator)
        else:
            z = posterior.mode()

        attention_mask = attention_masks[-1]
        logits = self.decode(z, attention_mask).sample

        if not return_dict:
            return (logits,)

        return AutoencoderKLOutput1D(sample=logits, latent_dist=posterior, attention_masks=attention_masks)
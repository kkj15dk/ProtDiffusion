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
from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn

from diffusers.utils import BaseOutput, is_torch_version
from diffusers.utils.torch_utils import randn_tensor
from diffusers.models.activations import get_activation
from diffusers.models.attention_processor import SpatialNorm
from .unet_1d_blocks import (
    UNetMidBlock1D,
    get_down_block,
    get_up_block,
    get_mid_block,
)

class DiagonalGaussianDistribution1D(object):
    def __init__(self, parameters: torch.Tensor, deterministic: bool = False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(
                self.mean, device=self.parameters.device, dtype=self.parameters.dtype
            )

    def sample(self, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        # make sure sample is on the same device as the parameters and has same dtype
        sample = randn_tensor(
            self.mean.shape,
            generator=generator,
            device=self.parameters.device,
            dtype=self.parameters.dtype,
        )
        x = self.mean + self.std * sample
        return x

    def kl(self) -> torch.Tensor:
        '''
        Compute the KL divergence between the distribution and the standard normal distribution.
        Using mean to reduce the KL divergence, so it does not depend on the batch and length dimension, which may vary.
        The channel dimension is the only dimension that stays constant.
        '''
        return 0.5 * torch.mean(
            torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar,
            dim = 1
        )

    def mode(self) -> torch.Tensor:
        return self.mean

@dataclass
class DecoderOutput(BaseOutput):
    r"""
    Output of decoding method.

    Args:
        sample (`torch.Tensor` of shape `(batch_size, num_channels, height, width)`):
            The decoded output sample from the last layer of the model.
    """

    sample: torch.Tensor
    kl: Optional[torch.FloatTensor] = None

@dataclass
class AutoencoderKLOutput1D(BaseOutput):
    """
    Output of AutoencoderKL encoding method.

    Args:
        latent_dist (`DiagonalGaussianDistribution`):
            Encoded outputs of `Encoder` represented as the mean and logvar of `DiagonalGaussianDistribution`.
            `DiagonalGaussianDistribution` allows for sampling latents from the distribution.
    """

    sample: torch.Tensor
    latent_dist: Optional["DiagonalGaussianDistribution1D"] = None
    attention_masks: Optional[List[torch.Tensor]] = None

@dataclass
class EncoderKLOutput1D(BaseOutput):
    """
    Output of AutoencoderKL encoding method.

    Args:
        latent_dist (`DiagonalGaussianDistribution`):
            Encoded outputs of `Encoder` represented as the mean and logvar of `DiagonalGaussianDistribution`.
            `DiagonalGaussianDistribution` allows for sampling latents from the distribution.
    """

    latent_dist: "DiagonalGaussianDistribution1D"  # noqa: F821
    attention_masks: List[torch.Tensor] = None

class Encoder1D(nn.Module):
    r"""
    The `Encoder` layer of a variational autoencoder that encodes its input into a latent representation.

    Args:
        in_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        out_channels (`int`, *optional*, defaults to 3):
            The number of output channels.
        down_block_types (`Tuple[str, ...]`, *optional*, defaults to `("DownEncoderBlock2D",)`):
            The types of down blocks to use. See `~diffusers.models.unet_2d_blocks.get_down_block` for available
            options.
        block_out_channels (`Tuple[int, ...]`, *optional*, defaults to `(64,)`):
            The number of output channels for each block.
        layers_per_block (`int`, *optional*, defaults to 2):
            The number of layers per block.
        norm_num_groups (`int`, *optional*, defaults to 32):
            The number of groups for normalization.
        act_fn (`str`, *optional*, defaults to `"silu"`):
            The activation function to use. See `~diffusers.models.activations.get_activation` for available options.
        double_z (`bool`, *optional*, defaults to `True`):
            Whether to double the number of output channels for the last block.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        down_block_types: Tuple[str, ...] = ("DownEncoderBlock1D",),
        mid_block_type: str = "UNetMidBlock1D",
        block_out_channels: Tuple[int, ...] = (64,),
        mid_block_channels: Optional[int] = None,
        layers_per_block: int = 2,
        transformer_layers_per_block: int = 1,
        norm_num_groups: int = 32,
        act_fn: str = "silu",
        norm_type: str = "group",  # group, spatial
        double_z: bool = True,
        mid_block_add_attention=True,
        num_attention_heads: int = 1,
    ):
        super().__init__()
        self.layers_per_block = layers_per_block

        self.conv_in = nn.Conv1d(
            in_channels,
            block_out_channels[0],
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.down_blocks = nn.ModuleList([])

        # down
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            down_block = get_down_block(
                down_block_type,
                num_layers=self.layers_per_block,
                transformer_layers_per_block=transformer_layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                add_downsample=True,
                resnet_eps=1e-6,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                num_attention_heads=num_attention_heads,
                temb_channels=None,
            )
            self.down_blocks.append(down_block)

        # mid
        if mid_block_channels is None:
            mid_block_channels = block_out_channels[-1]

        # mid
        self.mid_block = get_mid_block(
            mid_block_type=mid_block_type,
            temb_channels=None,
            in_channels=block_out_channels[-1],
            out_channels=mid_block_channels,
            num_layers=self.layers_per_block,
            transformer_layers_per_block=transformer_layers_per_block,
            resnet_eps=1e-6,
            resnet_act_fn=act_fn,
            output_scale_factor=1,
            resnet_time_scale_shift="default" if norm_type == "group" else norm_type,
            num_attention_heads=num_attention_heads,
            resnet_groups=norm_num_groups,
            add_attention=mid_block_add_attention,
        )

        # out
        self.conv_norm_out = nn.GroupNorm(num_channels=mid_block_channels, num_groups=norm_num_groups, eps=1e-6)
        self.conv_act = nn.SiLU()

        conv_out_channels = 2 * out_channels if double_z else out_channels
        self.conv_out = nn.Conv1d(mid_block_channels, conv_out_channels, 3, padding=1)

        self.gradient_checkpointing = False

    def forward(self, sample: torch.Tensor, attention_mask: torch.Tensor = None,) -> torch.Tensor:
        r"""The forward method of the `Encoder` class."""

        sample = self.conv_in(sample)

        attention_masks = [attention_mask]

        for down_block in self.down_blocks:
            sample, attention_mask = down_block(sample, attention_mask=attention_mask)
            attention_masks.append(attention_mask)

        # middle
        sample = self.mid_block(sample, attention_mask=attention_mask)

        # post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        return sample, attention_masks


class Decoder1D(nn.Module):
    r"""
    The `Decoder` layer of a variational autoencoder that decodes its latent representation into an output sample.

    Args:
        in_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        out_channels (`int`, *optional*, defaults to 3):
            The number of output channels.
        up_block_types (`Tuple[str, ...]`, *optional*, defaults to `("UpDecoderBlock2D",)`):
            The types of up blocks to use. See `~diffusers.models.unet_2d_blocks.get_up_block` for available options.
        block_out_channels (`Tuple[int, ...]`, *optional*, defaults to `(64,)`):
            The number of output channels for each block.
        layers_per_block (`int`, *optional*, defaults to 2):
            The number of layers per block.
        norm_num_groups (`int`, *optional*, defaults to 32):
            The number of groups for normalization.
        act_fn (`str`, *optional*, defaults to `"silu"`):
            The activation function to use. See `~diffusers.models.activations.get_activation` for available options.
        norm_type (`str`, *optional*, defaults to `"group"`):
            The normalization type to use. Can be either `"group"` or `"spatial"`.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        up_block_types: Tuple[str, ...] = ("UpDecoderBlock1D",),
        mid_block_type: str = "UNetMidBlock1D",
        block_out_channels: Tuple[int, ...] = (64,),
        mid_block_channels: Optional[int] = None,
        layers_per_block: int = 2,
        transformer_layers_per_block: int = 1,
        norm_num_groups: int = 32,
        act_fn: str = "silu",
        norm_type: str = "group",  # group, spatial
        mid_block_add_attention=True,
        num_attention_heads: int = 1,
    ):
        super().__init__()
        self.layers_per_block = layers_per_block

        if mid_block_channels is None:
            mid_block_channels = block_out_channels[-1]

        self.conv_in = nn.Conv1d(
            in_channels,
            mid_block_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.up_blocks = nn.ModuleList([])

        temb_channels = in_channels if norm_type == "spatial" else None

        # mid
        self.mid_block = get_mid_block(
            mid_block_type=mid_block_type,
            temb_channels=None,
            in_channels=mid_block_channels,
            out_channels=mid_block_channels,
            num_layers=self.layers_per_block,
            transformer_layers_per_block=transformer_layers_per_block,
            resnet_eps=1e-6,
            resnet_act_fn=act_fn,
            output_scale_factor=1,
            resnet_time_scale_shift="default" if norm_type == "group" else norm_type,
            num_attention_heads=num_attention_heads,
            resnet_groups=norm_num_groups,
            add_attention=mid_block_add_attention,
        )

        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = mid_block_channels
        for i, up_block_type in enumerate(up_block_types):

            input_channel = output_channel
            output_channel = reversed_block_out_channels[i]

            up_block = get_up_block(
                up_block_type,
                num_layers=self.layers_per_block,
                transformer_layers_per_block=transformer_layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                add_upsample=True,
                resnet_eps=1e-6,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                num_attention_heads=num_attention_heads,
                temb_channels=None,
                resnet_time_scale_shift=norm_type,
            )
            self.up_blocks.append(up_block)

        # out
        if norm_type == "spatial":
            self.conv_norm_out = SpatialNorm(block_out_channels[0], temb_channels)
        else:
            self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[0], num_groups=norm_num_groups, eps=1e-6)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv1d(block_out_channels[0], out_channels, 3, padding=1)

        self.gradient_checkpointing = False

    def forward(
        self,
        sample: torch.Tensor,
        attention_masks: List[torch.Tensor] = None,
    ) -> torch.Tensor:
        r"""The forward method of the `Decoder` class."""

        sample = self.conv_in(sample)

        upscale_dtype = next(iter(self.up_blocks.parameters())).dtype

        # middle
        sample = self.mid_block(sample, attention_mask=attention_masks[-1])
        sample = sample.to(upscale_dtype)

        # up
        for i, up_block in enumerate(self.up_blocks):
            sample = up_block(sample, attention_mask=attention_masks[-(i + 2)])

        # post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        return sample
# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional, Sequence, Tuple, Type, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.networks.blocks.dynunet_block import UnetResBlock, get_conv_layer
from monai.networks.layers.utils import get_act_layer, get_norm_layer


class GcisBasicBlock(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[Sequence[int], int],
        stride: Union[Sequence[int], int],
        norm_name: Union[Tuple, str],
        bottle_neck: bool = False,
    ) -> None:

        super().__init__()
    
        if bottle_neck:
            self.layer = BottleNeckResBlock(
                spatial_dims,
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                norm_name=norm_name,
            )
        else:
            self.layer = UnetResBlock(
                spatial_dims,
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                norm_name=norm_name,
            )

    def forward(self, inp):
        return self.layer(inp)

class GcisApBasicBlock(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[Sequence[int], int],
        stride: Union[Sequence[int], int],
        norm_name: Union[Tuple, str],
        ap_num: int,
        bottle_neck: bool = False,
    ) -> None:

        super().__init__()
    
        self.ap_num = ap_num
        self.prompt_router = nn.Linear(512, ap_num)
        if bottle_neck:
            for i in range(ap_num):
                exec("self.layer_p{} = BottleNeckResBlock(spatial_dims, in_channels, out_channels, kernel_size=kernel_size, stride=stride, norm_name=norm_name,)".format(i))
        else:
            for i in range(ap_num):
                exec("self.layer_p{} = UnetResBlock(spatial_dims, in_channels, out_channels, kernel_size=kernel_size, stride=stride, norm_name=norm_name,)".format(i))

    def forward(self, inp, prompt_in, tau, layer_name=None, display_path=False):
        prompt_select = self.prompt_router(prompt_in)
        if self.training:
            prompt_select = F.log_softmax(prompt_select, dim=1)
            prompt_select = F.gumbel_softmax(prompt_select, tau=tau, hard=True)
        else:
            prompt_select = torch.softmax(prompt_select, dim=1)
            index = prompt_select.max(1, keepdim=True)[1]
            if display_path:
                print(layer_name + " path index: ", int(index[0][0])+1)
            prompt_select = torch.zeros_like(prompt_select).scatter_(1, index, 1.0)
        b, c, _, _, _ = inp.size()
        final_out = []
        for i in range(b):
            out_cur = inp[i, :, :, :, :].unsqueeze(0)
            connect_out = 0
            for path_idx in range(self.ap_num):
                path_out = eval("self.layer_p{}(out_cur)".format(path_idx))
                connect_out += path_out * prompt_select[i, path_idx]
            final_out.append(connect_out)
        final_out = torch.cat(final_out, 0)
        return final_out

class GcisApUpBlock(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[Sequence[int], int],
        upsample_kernel_size: Union[Sequence[int], int],
        norm_name: Union[Tuple, str],
        ap_num: int,
        bottle_neck: bool = False,
    ) -> None:

        super().__init__()
        upsample_stride = upsample_kernel_size
        self.transp_conv = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=upsample_kernel_size,
            stride=upsample_stride,
            conv_only=True,
            is_transposed=True,
        )

        self.concat_conv = get_conv_layer(
                            spatial_dims,
                            out_channels + out_channels,
                            out_channels,
                            kernel_size=1,
                            stride=1,
                            dropout=None,
                            act=None,
                            norm=None,
                            conv_only=False,
                        )
        self.norm1 = get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=out_channels)
        self.lrelu = get_act_layer(name=("leakyrelu", {"inplace": True, "negative_slope": 0.01}))

        self.ap_num = ap_num
        self.prompt_router = nn.Linear(512, ap_num)
        if bottle_neck:
            for i in range(ap_num):
                exec("self.conv_block_p{} = BottleNeckResBlock(spatial_dims, out_channels, out_channels, kernel_size=kernel_size, stride=1, norm_name=norm_name,)".format(i))
        else:
            for i in range(ap_num):
                exec("self.conv_block_p{} = UnetResBlock(spatial_dims, out_channels, out_channels, kernel_size=kernel_size, stride=1, norm_name=norm_name,)".format(i))

    def forward(self, inp, skip, prompt_in, tau, layer_name=None, display_path=False):
        out = self.transp_conv(inp)
        out = torch.cat((out, skip), dim=1)
        out = self.concat_conv(out)
        out = self.norm1(out)
        out = self.lrelu(out)
        prompt_select = self.prompt_router(prompt_in)
        if self.training:
            prompt_select = F.log_softmax(prompt_select, dim=1)
            prompt_select = F.gumbel_softmax(prompt_select, tau=tau, hard=True)
        else:
            prompt_select = torch.softmax(prompt_select, dim=1)
            index = prompt_select.max(1, keepdim=True)[1]
            if display_path:
                print(layer_name + " path index: ", int(index[0][0])+1)
            prompt_select = torch.zeros_like(prompt_select).scatter_(1, index, 1.0)
        b, c, _, _, _ = out.size()
        final_out = []
        for i in range(b):
            out_cur = out[i, :, :, :, :].unsqueeze(0)
            connect_out = 0
            for path_idx in range(self.ap_num):
                path_out = eval("self.conv_block_p{}(out_cur)".format(path_idx))
                connect_out += path_out * prompt_select[i, path_idx]
            final_out.append(connect_out)
        final_out = torch.cat(final_out, 0)
        return final_out

class GcisUpBlock(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[Sequence[int], int],
        upsample_kernel_size: Union[Sequence[int], int],
        norm_name: Union[Tuple, str],
        bottle_neck: bool = False,
    ) -> None:

        super().__init__()
        upsample_stride = upsample_kernel_size
        self.transp_conv = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=upsample_kernel_size,
            stride=upsample_stride,
            conv_only=True,
            is_transposed=True,
        )

        self.concat_conv = get_conv_layer(
                            spatial_dims,
                            out_channels + out_channels,
                            out_channels,
                            kernel_size=1,
                            stride=1,
                            dropout=None,
                            act=None,
                            norm=None,
                            conv_only=False,
                        )
        self.norm1 = get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=out_channels)
        self.lrelu = get_act_layer(name=("leakyrelu", {"inplace": True, "negative_slope": 0.01}))

        if bottle_neck:
            self.conv_block = BottleNeckResBlock(spatial_dims, out_channels, out_channels, kernel_size=kernel_size, stride=1, norm_name=norm_name,)
        else:
            self.conv_block = UnetResBlock(spatial_dims, out_channels, out_channels, kernel_size=kernel_size, stride=1, norm_name=norm_name,)

    def forward(self, inp, skip):
        out = self.transp_conv(inp)
        out = torch.cat((out, skip), dim=1)
        out = self.concat_conv(out)
        out = self.norm1(out)
        out = self.lrelu(out)
        out = self.conv_block(out)
        return out


class BottleNeckResBlock(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[Sequence[int], int],
        stride: Union[Sequence[int], int],
        norm_name: Union[Tuple, str],
        act_name: Union[Tuple, str] = ("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
        dropout: Optional[Union[Tuple, str, float]] = None,
    ):
        super().__init__()
        r = 2
        self.lrelu = get_act_layer(name=act_name)
        self.conv1 = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels//r,
            kernel_size=1,
            stride=stride,
            dropout=dropout,
            act=None,
            norm=None,
            conv_only=False,
        )
        self.norm1 = get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=out_channels//r)
        self.conv2 = get_conv_layer(
            spatial_dims,
            out_channels//r,
            out_channels//r,
            kernel_size=kernel_size,
            stride=1,
            dropout=dropout,
            act=None,
            norm=None,
            conv_only=False,
        )
        self.norm2 = get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=out_channels//r)
        self.conv2_2 = get_conv_layer(
            spatial_dims,
            out_channels//r,
            out_channels,
            kernel_size=1,
            stride=1,
            dropout=dropout,
            act=None,
            norm=None,
            conv_only=False,
        )
        self.norm2_2 = get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=out_channels)
        self.downsample = in_channels != out_channels
        stride_np = np.atleast_1d(stride)
        if not np.all(stride_np == 1):
            self.downsample = True
        if self.downsample:
            self.conv3 = get_conv_layer(
                spatial_dims,
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                dropout=dropout,
                act=None,
                norm=None,
                conv_only=False,
            )
            self.norm3 = get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=out_channels)

    def forward(self, inp):
        residual = inp
        out = self.conv1(inp)
        out = self.norm1(out)
        out = self.lrelu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.lrelu(out)
        out = self.conv2_2(out)
        out = self.norm2_2(out)
        if hasattr(self, "conv3"):
            residual = self.conv3(residual)
        if hasattr(self, "norm3"):
            residual = self.norm3(residual)
        out += residual
        out = self.lrelu(out)
        return out
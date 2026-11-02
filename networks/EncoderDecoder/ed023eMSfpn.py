# Copyright 2020 University of Basel, Center for medical Image Analysis and Navigation
#
# Licensed under the Apache License, Version 2.0 (the “License”);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an “AS IS” BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ed023eMSfpn: ed023eMS with FPN-style lateral injection into the decode path. The output
# heads are unchanged:
#   after conv5 -> 1/4 res (4*nf ch) -> conv7_64  -> 'out64'
#   after conv6 -> 1/2 res (2*nf ch) -> conv7_128 -> 'out128'
#   after up1   -> full res (nf ch)  -> conv7_k   -> 'out0'
# but decode accepts `inject`, a dict of channel-matched 3D feature tensors added to the
# trunk AFTER each coarse head's tap:
#   inject['pre_up2'] (4*nf ch) added after the out64 tap, before up2
#   inject['pre_up1'] (2*nf ch) added after the out128 tap, before up1
# so out64 sees only the trunk input (coarse VQ scales), out128 additionally sees
# 'pre_up2', and only out0 sees everything. Injected tensors may be at any spatial size
# (typically the latent resolution); they are trilinearly resized to the trunk feature size
# at the injection point. The adapters projecting VQ scales to 4*nf / 2*nf channels live in
# the model (models/vqcleanM0aMSfpn.py), keeping this file free of VQ specifics.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from networks.model_utils import get_activation

sig = nn.Sigmoid()
ACTIVATION = nn.ReLU
#device = 'cuda'


class Flatten(torch.nn.Module):
    def forward(self, x):
        return x.reshape(x.size()[0], -1)

def crop_and_concat(upsampled, bypass, crop=False):
    if crop:
        c = (bypass.size()[2] - upsampled.size()[2]) // 2
        bypass = F.pad(bypass, (-c, -c, -c, -c))

    return torch.cat((upsampled, bypass), 1)



def get_normalization(out_channels, method):
    if method == 'batch':
        return nn.BatchNorm2d(out_channels)
    elif method == 'instance':
        return nn.InstanceNorm2d(out_channels)
    elif method == 'group':
        return nn.GroupNorm(32, out_channels)
    elif method == 'none':
        return nn.Identity()

def get_normalization_3d(out_channels, method):
    if method == 'batch':
        return nn.BatchNorm3d(out_channels)
    elif method == 'instance':
        return nn.InstanceNorm3d(out_channels)
    elif method == 'group':
        return nn.GroupNorm(32, out_channels)
    elif method == 'none':
        return nn.Identity()


def conv2d_bn_block(in_channels, out_channels, kernel=3, momentum=0.01, activation=ACTIVATION, norm='batch'):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=1),
        #nn.BatchNorm2d(out_channels, momentum=momentum),
        get_normalization(out_channels, method=norm),
        activation(),
    )


def deconv2d_bn_block(in_channels, out_channels, use_upsample=True, kernel=4, stride=2, padding=1, momentum=0.01,
                      activation=ACTIVATION, norm='batch'):
    if use_upsample:
        up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1)
        )
    else:
        up = nn.ConvTranspose2d(in_channels, out_channels, kernel, stride=stride, padding=padding)
    return nn.Sequential(
        up,
        #nn.BatchNorm2d(out_channels, momentum=momentum),
        get_normalization(out_channels, method=norm),
        activation(),
    )


def conv3d_bn_block(in_channels, out_channels, kernel=3, momentum=0.01, activation=ACTIVATION, norm='batch'):
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel, padding=1),
        #nn.BatchNorm3d(out_channels, momentum=momentum),
        get_normalization_3d(out_channels, method=norm),
        activation(),
    )


class NearestUpsample3d(nn.Module):
    """nn.Upsample(nearest) that chunks along channels when the output would exceed
    the int32 element limit of the CUDA upsample_nearest3d kernel — hit on full-volume
    validation (e.g. 2*nf ch x 384^3 at up1). Parameter-free, so state_dict-compatible
    with nn.Upsample."""
    INT32_MAX = 2 ** 31 - 1

    def __init__(self, scale_factor):
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        s = self.scale_factor
        sy, sx, sz = s if isinstance(s, (tuple, list)) else (s, s, s)
        out_shape = (x.shape[0], x.shape[1],
                     int(x.shape[2] * sy), int(x.shape[3] * sx), int(x.shape[4] * sz))
        out_numel = out_shape[0] * out_shape[1] * out_shape[2] * out_shape[3] * out_shape[4]
        if out_numel <= self.INT32_MAX:
            return nn.functional.interpolate(x, scale_factor=self.scale_factor)
        per_ch = out_numel // out_shape[1]
        step = max(1, self.INT32_MAX // per_ch)
        out = x.new_empty(out_shape)
        for c0 in range(0, out_shape[1], step):
            out[:, c0:c0 + step] = nn.functional.interpolate(
                x[:, c0:c0 + step], scale_factor=self.scale_factor)
        return out


class SafeUpConv3d(nn.Sequential):
    """Nearest-Upsample -> Conv3d(k3, p1) pair that switches to exact Z-slab chunking
    (1-voxel conv halo) when the upsampled tensor would exceed cuDNN's int32 element
    limit — beyond it F.conv3d falls back to an im2col path needing a ~360 GiB
    workspace (hit on full-volume validation, e.g. 2*nf ch x 384^3 at up1).
    Subclasses nn.Sequential so state_dict keys match the plain (Upsample, Conv3d)
    pair. Chunking is only ever entered at inference-scale inputs."""
    SAFE_NUMEL = 2 ** 30  # half the int32 kernel limit, margin for cuDNN workspaces

    def forward(self, x):
        up, conv = self[0], self[1]
        s = up.scale_factor
        sy, sx, sz = s if isinstance(s, (tuple, list)) else (s, s, s)
        if x.numel() * sy * sx * sz <= self.SAFE_NUMEL:
            return conv(up(x))
        zin = x.shape[4]
        # elements of one upsampled z-plane -> input-z rows per chunk
        plane = x.shape[0] * x.shape[1] * int(x.shape[2] * sy) * int(x.shape[3] * sx)
        step = max(1, int(self.SAFE_NUMEL // plane - 2) // int(sz))
        outs = []
        for a in range(0, zin, step):
            b = min(a + step, zin)
            a2, b2 = a - (1 if a > 0 else 0), b + (1 if b < zin else 0)  # halo rows
            slab = nn.functional.interpolate(x[..., a2:b2], scale_factor=(sy, sx, sz))
            # crop to the needed upsampled range [a*sz - halo, b*sz + halo)
            u0 = a * int(sz) - (1 if a > 0 else 0)
            u1 = b * int(sz) + (1 if b < zin else 0)
            slab = slab[..., u0 - a2 * int(sz):u1 - a2 * int(sz)]
            # zero-pad Z only at the global volume ends; interior edges use halo rows
            slab = nn.functional.pad(slab, (1 if a == 0 else 0, 1 if b == zin else 0))
            outs.append(nn.functional.conv3d(slab, conv.weight, conv.bias,
                                             stride=1, padding=(1, 1, 0)))
        return torch.cat(outs, dim=4)


def deconv3d_bn_block(in_channels, out_channels, use_upsample=(2, 2, 2), kernel=4, stride=2, padding=1, momentum=0.01,
                      activation=ACTIVATION, norm='batch'):
    if use_upsample:
        up = SafeUpConv3d(
            NearestUpsample3d(scale_factor=use_upsample),
            nn.Conv3d(in_channels, out_channels, 3, stride=1, padding=1)
        )
    else:
        up = nn.ConvTranspose3d(in_channels, out_channels, kernel, stride=stride, padding=padding)
    return nn.Sequential(
        up,
        #nn.BatchNorm3d(out_channels, momentum=momentum),
        get_normalization_3d(out_channels, method=norm),
        activation(),
    )


class Generator(nn.Module):
    def __init__(self, n_channels=1, out_channels=1, nf=32, norm_type='batch', activation=ACTIVATION, final='tanh', mc=False, use_upsample=False):
        super(Generator, self).__init__()

        if norm_type == 'batch':
            batch_norm = True

        conv2_block = conv2d_bn_block
        conv3_block = conv3d_bn_block

        max2_pool = nn.MaxPool2d(2)
        self.max3_pool = nn.MaxPool3d(2)
        act = activation

        if mc:
            dropout = 0.5
        else:
            dropout = 0.0

        self.down0 = nn.Sequential(
            conv2_block(n_channels, nf, activation=act, norm='none'),
            conv2_block(nf, nf, activation=act, norm=norm_type)
        )
        self.down1 = nn.Sequential(
            conv2_block(nf, 2 * nf, activation=act, norm=norm_type),
            conv2_block(2 * nf, 2 * nf, activation=act, norm=norm_type),
        )
        self.down2 = nn.Sequential(
            conv2_block(2 * nf, 4 * nf, activation=act, norm=norm_type),
            nn.Dropout(p=dropout, inplace=False),
            conv2_block(4 * nf, 4 * nf, activation=act, norm=norm_type),

        )
        self.down3 = nn.Sequential(
            conv2_block(4 * nf, 8 * nf, activation=act, norm=norm_type),
            nn.Dropout(p=dropout, inplace=False),
            conv2_block(8 * nf, 8 * nf, activation=act, norm=norm_type),
        )

        self.up3 = deconv3d_bn_block(8 * nf, 4 * nf, activation=act, norm=norm_type, use_upsample=use_upsample)

        self.conv5 = nn.Sequential(
            conv3_block(4 * nf, 4 * nf, activation=act, norm=norm_type),  # 8
            nn.Dropout(p=dropout, inplace=False),
            conv3_block(4 * nf, 4 * nf, activation=act, norm=norm_type),
        )
        self.up2 = deconv3d_bn_block(4 * nf, 2 * nf, activation=act, norm=norm_type, use_upsample=use_upsample)
        self.conv6 = nn.Sequential(
            conv3_block(2 * nf, 2 * nf, activation=act, norm=norm_type),
            nn.Dropout(p=dropout, inplace=False),
            conv3_block(2 * nf, 2 * nf, activation=act, norm=norm_type),
        )

        self.up1 = deconv3d_bn_block(2 * nf, nf, activation=act, norm=norm_type, use_upsample=use_upsample)

        final_layer = get_activation(final)

        self.conv7_k = nn.Sequential(
            conv3_block(nf, out_channels, activation=final_layer, norm='none'),
        )

        self.conv7_g = nn.Sequential(
            conv3_block(nf, out_channels, activation=final_layer, norm='none'),
        )

        # Coarse-resolution output heads (progressive decode).
        # conv7_128 taps after conv6 (1/2 res, 2*nf ch); conv7_64 taps after conv5 (1/4 res, 4*nf ch).
        self.conv7_128 = nn.Sequential(
            conv3_block(2 * nf, out_channels, activation=final_layer, norm='none'),
        )
        self.conv7_64 = nn.Sequential(
            conv3_block(4 * nf, out_channels, activation=final_layer, norm='none'),
        )

        self.encoder = nn.Sequential(self.down0, self.down1, self.down2, self.down3)
        self.decoder = nn.Sequential(self.up3, self.conv5, self.up2, self.conv6, self.up1)

    def forward(self, x, method=None, inject=None, stop_at=None):
        # x (B, C, Y, X, Z) — repo layout: dim2 = Y, dim3 = X, dim4 = Z
        # stop_at: 'out64' | 'out128' — early-exit right after that head's tap,
        # skipping the deeper (and much more expensive) up2/up1 stages. The
        # returned dict then only contains the heads computed so far. The taps
        # sit BEFORE the lateral injections, so early-exit outputs are
        # bit-identical to the same keys of a full pass.
        if method != 'decode':
            x = x.permute(4, 1, 2, 3, 0)[:, :, :, :, 0]  # (Z, C, Y, X); encode path keeps batch item 0 only
            feat = []
            for i in range(len(self.encoder)):
                if i > 0:
                    x = x.permute(1, 2, 3, 0).unsqueeze(0)  # (1, C, Y, X, Z)
                    x = self.max3_pool(x)
                    x = x.squeeze(0).permute(3, 0, 1, 2)  # (Z, C, Y, X)
                x = self.encoder[i](x)
                feat.append(x.permute(1, 2, 3, 0).unsqueeze(0))
            if method == 'encode':
                return feat
            x = x.permute(1, 2, 3, 0).unsqueeze(0)  # (1, C, Y, X, Z)

        # Explicit decode stages so intermediate resolutions can be tapped by extra heads.
        # Lateral injections land AFTER each coarse head's tap, restricting what it can see.
        x = self.up3(x)
        x = self.conv5(x)          # 1/4 res, 4*nf channels
        out64 = self.conv7_64(x)
        if stop_at == 'out64':
            return {'out64': out64}
        if inject is not None and 'pre_up2' in inject:
            x = x + F.interpolate(inject['pre_up2'], size=x.shape[2:],
                                  mode='trilinear', align_corners=False)
        x = self.up2(x)
        x = self.conv6(x)          # 1/2 res, 2*nf channels
        out128 = self.conv7_128(x)
        if stop_at == 'out128':
            return {'out128': out128, 'out64': out64}
        if inject is not None and 'pre_up1' in inject:
            x = x + F.interpolate(inject['pre_up1'], size=x.shape[2:],
                                  mode='trilinear', align_corners=False)
        x = self.up1(x)            # full res, nf channels
        x70 = self.conv7_k(x)
        x71 = self.conv7_g(x)

        return {'out0': x70, 'out1': x71, 'out128': out128, 'out64': out64}


if __name__ == '__main__':
    g = Generator(n_channels=1, norm_type='group', final='tanh', use_upsample=False)
    f = g(torch.rand(1, 1, 128, 128, 128), method='encode')
    print(f[-1].shape)
    out = g(f[-1], method='decode')
    print({k: tuple(v.shape) for k, v in out.items()})

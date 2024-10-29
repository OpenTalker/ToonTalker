from audioop import bias
import math
from turtle import forward
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from torch.nn.utils.spectral_norm import spectral_norm as SpectralNorm
from .transformer import TransformerDecoderLayer

def fused_leaky_relu(input, bias, negative_slope=0.2, scale=2 ** 0.5):
    return F.leaky_relu(input + bias, negative_slope) * scale


class FusedLeakyReLU(nn.Module):
    def __init__(self, channel, negative_slope=0.2, scale=2 ** 0.5):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1, channel, 1, 1))
        self.negative_slope = negative_slope
        self.scale = scale

    def forward(self, input):
        # print("FusedLeakyReLU: ", input.abs().mean())
        out = fused_leaky_relu(input, self.bias, self.negative_slope, self.scale)
        # print("FusedLeakyReLU: ", out.abs().mean())
        return out


def upfirdn2d_native(input, kernel, up_x, up_y, down_x, down_y, pad_x0, pad_x1, pad_y0, pad_y1):
    _, minor, in_h, in_w = input.shape
    kernel_h, kernel_w = kernel.shape

    out = input.view(-1, minor, in_h, 1, in_w, 1)
    out = F.pad(out, [0, up_x - 1, 0, 0, 0, up_y - 1, 0, 0])
    out = out.view(-1, minor, in_h * up_y, in_w * up_x)

    out = F.pad(out, [max(pad_x0, 0), max(pad_x1, 0), max(pad_y0, 0), max(pad_y1, 0)])
    out = out[:, :, max(-pad_y0, 0): out.shape[2] - max(-pad_y1, 0),
          max(-pad_x0, 0): out.shape[3] - max(-pad_x1, 0), ]

    # out = out.permute(0, 3, 1, 2)
    out = out.reshape([-1, 1, in_h * up_y + pad_y0 + pad_y1, in_w * up_x + pad_x0 + pad_x1])
    w = torch.flip(kernel, [0, 1]).view(1, 1, kernel_h, kernel_w)
    out = F.conv2d(out, w)
    out = out.reshape(-1, minor, in_h * up_y + pad_y0 + pad_y1 - kernel_h + 1,
                      in_w * up_x + pad_x0 + pad_x1 - kernel_w + 1, )
    # out = out.permute(0, 2, 3, 1)

    return out[:, :, ::down_y, ::down_x]


def upfirdn2d(input, kernel, up=1, down=1, pad=(0, 0)):
    return upfirdn2d_native(input, kernel, up, up, down, down, pad[0], pad[1], pad[0], pad[1])

def make_kernel(k):
    k = torch.tensor(k, dtype=torch.float32)

    if k.ndim == 1:
        k = k[None, :] * k[:, None]

    k /= k.sum()

    return k

class Upsample(nn.Module):
    def __init__(self, kernel, factor=2):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel) * (factor ** 2)
        self.register_buffer('kernel', kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2 + factor - 1
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def forward(self, input):
        return upfirdn2d(input, self.kernel, up=self.factor, down=1, pad=self.pad)


class Downsample(nn.Module):
    def __init__(self, kernel, factor=2):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel)
        self.register_buffer('kernel', kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def forward(self, input):
        return upfirdn2d(input, self.kernel, up=1, down=self.factor, pad=self.pad)


class Blur(nn.Module):
    def __init__(self, kernel, pad, upsample_factor=1):
        super().__init__()

        kernel = make_kernel(kernel)

        if upsample_factor > 1:
            kernel = kernel * (upsample_factor ** 2)

        self.register_buffer('kernel', kernel)

        self.pad = pad

    def forward(self, input):
        return upfirdn2d(input, self.kernel, pad=self.pad)


class EqualConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_channel, in_channel, kernel_size, kernel_size))
        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)

        self.stride = stride
        self.padding = padding

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel))
        else:
            self.bias = None

    def forward(self, input):

        return F.conv2d(input, self.weight * self.scale, bias=self.bias, stride=self.stride, padding=self.padding, )

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]},'
            f' {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})'
        )

class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1, activation=None):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))
        else:
            self.bias = None

        self.activation = activation

        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input):

        if self.activation:
            out = F.linear(input, self.weight * self.scale)
            out = fused_leaky_relu(out, self.bias * self.lr_mul)
        else:
            out = F.linear(input, self.weight * self.scale, bias=self.bias * self.lr_mul)

        return out

    def __repr__(self):
        return (f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})')


class ScaledLeakyReLU(nn.Module):
    def __init__(self, negative_slope=0.2):
        super().__init__()

        self.negative_slope = negative_slope

    def forward(self, input):
        return F.leaky_relu(input, negative_slope=self.negative_slope)


class ModulatedConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, style_dim, demodulate=True, upsample=False,
                 downsample=False, blur_kernel=[1, 3, 3, 1], ):
        super().__init__()

        self.eps = 1e-8
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.upsample = upsample
        self.downsample = downsample

        if upsample:
            factor = 2
            p = (len(blur_kernel) - factor) - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1

            self.blur = Blur(blur_kernel, pad=(pad0, pad1), upsample_factor=factor)

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            self.blur = Blur(blur_kernel, pad=(pad0, pad1))

        fan_in = in_channel * kernel_size ** 2
        self.scale = 1 / math.sqrt(fan_in)
        self.padding = kernel_size // 2

        self.weight = nn.Parameter(torch.randn(1, out_channel, in_channel, kernel_size, kernel_size))

        self.modulation = EqualLinear(style_dim, in_channel, bias_init=1)
        self.demodulate = demodulate

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.in_channel}, {self.out_channel}, {self.kernel_size}, '
            f'upsample={self.upsample}, downsample={self.downsample})'
        )

    def forward(self, input, style):
        batch, in_channel, height, width = input.shape

        style = self.modulation(style).view(batch, 1, in_channel, 1, 1)
        weight = self.scale * self.weight * style

        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
            weight = weight * demod.view(batch, self.out_channel, 1, 1, 1)

        weight = weight.view(batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size)

        if self.upsample:
            input = input.view(1, batch * in_channel, height, width)
            weight = weight.view(batch, self.out_channel, in_channel, self.kernel_size, self.kernel_size)
            weight = weight.transpose(1, 2).reshape(batch * in_channel, self.out_channel, self.kernel_size,
                                                    self.kernel_size)
            out = F.conv_transpose2d(input, weight, padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)
            out = self.blur(out)
        elif self.downsample:
            input = self.blur(input)
            _, _, height, width = input.shape
            input = input.view(1, batch * in_channel, height, width)
            out = F.conv2d(input, weight, padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)
        else:
            input = input.view(1, batch * in_channel, height, width)
            out = F.conv2d(input, weight, padding=self.padding, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        return out


class NoiseInjection(nn.Module):
    def __init__(self):
        super().__init__()

        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, image, noise=None):

        if noise is None:
            return image
        else:
            return image + self.weight * noise


class ConstantInput(nn.Module):
    def __init__(self, channel, size=4):
        super().__init__()

        self.input = nn.Parameter(torch.randn(1, channel, size, size))

    def forward(self, input):
        batch = input.shape[0]
        out = self.input.repeat(batch, 1, 1, 1)

        return out


class StyledConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, style_dim, upsample=False, blur_kernel=[1, 3, 3, 1],
                 demodulate=True):
        super().__init__()

        self.conv = ModulatedConv2d(
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
            upsample=upsample,
            blur_kernel=blur_kernel,
            demodulate=demodulate,
        )

        self.noise = NoiseInjection()
        self.activate = FusedLeakyReLU(out_channel)

    def forward(self, input, style, noise=None):
        out = self.conv(input, style)
        out = self.noise(out, noise=noise)
        out = self.activate(out)

        return out

def spectral_norm(module, use_spect=True):
    """use spectral normal layer to stable the training process"""
    if use_spect:
        return SpectralNorm(module)
    else:
        return module

class ConvLayer(nn.Sequential):
    def __init__(
            self,
            in_channel,
            out_channel,
            kernel_size,
            downsample=False,
            blur_kernel=[1, 3, 3, 1],
            bias=True,
            activate=True,
    ):
        layers = []

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            layers.append(Blur(blur_kernel, pad=(pad0, pad1)))

            stride = 2
            self.padding = 0

        else:
            stride = 1
            self.padding = kernel_size // 2

        layers.append(EqualConv2d(in_channel, out_channel, kernel_size, padding=self.padding, stride=stride,
                                  bias=bias and not activate))

        if activate:
            if bias:
                layers.append(FusedLeakyReLU(out_channel))
            else:
                layers.append(ScaledLeakyReLU(0.2))

        super().__init__(*layers)


class ToRGB(nn.Module):
    def __init__(self, in_channel, style_dim, upsample=True, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        if upsample:
            self.upsample = Upsample(blur_kernel)

        self.conv = ConvLayer(in_channel, 3, 1)
        self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))

    def forward(self, input, skip=None):
        out = self.conv(input)
        out = out + self.bias

        if skip is not None:
            skip = self.upsample(skip)
            out = out + skip

        return out


class ToFlow(nn.Module):
    def __init__(self, in_channel, style_dim, upsample=True, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        if upsample:
            self.upsample = Upsample(blur_kernel)

        self.conv = ModulatedConv2d(in_channel, 3, 1, style_dim, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))

    def forward(self, input, style, feat, skip=None, sft = False):
        # print(feat.shape, input.shape)
        out = self.conv(input, style)
        out = out + self.bias
        # warping
        xs = np.linspace(-1, 1, input.size(2))
        xs = np.meshgrid(xs, xs)
        xs = np.stack(xs, 2)

        xs = torch.tensor(xs, requires_grad=False).float().unsqueeze(0).repeat(input.size(0), 1, 1, 1).cuda()

        if skip is not None:
            skip = self.upsample(skip)
            out = out + skip

        sampler = torch.tanh(out[:, 0:2, :, :])
        mask = torch.sigmoid(out[:, 2:3, :, :])
        flow = sampler.permute(0, 2, 3, 1) + xs

        feat_warp = F.grid_sample(feat, flow) * mask
        # print(feat_warp.shape, input.shape)
        if sft:
            return feat_warp, mask, out
        else:
            return feat_warp, feat_warp + input * (1.0 - mask), out
    
    def validation(self, input, style, feat, skip=None):
        out = self.conv(input, style)
        out = out + self.bias

        # warping
        xs = np.linspace(-1, 1, input.size(2))
        xs = np.meshgrid(xs, xs)
        xs = np.stack(xs, 2)

        xs = torch.tensor(xs, requires_grad=False).float().unsqueeze(0).repeat(input.size(0), 1, 1, 1).cuda()

        if skip is not None:
            skip = self.upsample(skip)
            out = out + skip

        sampler = torch.tanh(out[:, 0:2, :, :])
        mask = torch.sigmoid(out[:, 2:3, :, :])
        flow = sampler.permute(0, 2, 3, 1) + xs

        feat_warp = F.grid_sample(feat, flow) * mask

        return feat_warp, feat_warp + input * (1.0 - mask), out, flow

    def visualize(self, input, style, feat, skip=None):
        out = self.conv(input, style)
        out = out + self.bias
        # warping
        xs = np.linspace(-1, 1, input.size(2))
        xs = np.meshgrid(xs, xs)
        xs = np.stack(xs, 2)

        xs = torch.tensor(xs, requires_grad=False).float().unsqueeze(0).repeat(input.size(0), 1, 1, 1).cuda()

        if skip is not None:
            skip = self.upsample(skip)
            out = out + skip

        sampler = torch.tanh(out[:, 0:2, :, :])
        mask = torch.sigmoid(out[:, 2:3, :, :])
        flow = sampler.permute(0, 2, 3, 1) + xs

        feat_warp = F.grid_sample(feat, flow) * mask

        return feat_warp, feat_warp + input * (1.0 - mask), out, sampler

class Direction(nn.Module):
    def __init__(self, motion_dim):
        super(Direction, self).__init__()

        self.weight = nn.Parameter(torch.randn(512, motion_dim))

    def forward(self, input):
        # input: (bs*t) x 512

        weight = self.weight + 1e-8
        Q, R = torch.linalg.qr(weight) #torch.qr(weight)  # get eignvector, orthogonal [n1, n2, n3, n4]

        if input is None:
            return Q
        else:
            input_diag = torch.diag_embed(input)  # alpha, diagonal matrix
            out = torch.matmul(input_diag, Q.T)
            out = torch.sum(out, dim=1)

            return out

class Fuse_sft_block(nn.Module):
    def __init__(self, in_ch, out_ch, style_dim):
        super().__init__()

        self.scale1 = StyledConv(in_ch, out_ch, 3, style_dim)
        self.scale2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.act1 = nn.LeakyReLU(0.2, True)
        
        self.shift1 = StyledConv(in_ch, out_ch, 3, style_dim)
        self.shift2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.act2 = nn.LeakyReLU(0.2, True)

    def forward(self, feat, style):
        scale = self.scale1(feat, style)
        scale = self.scale2(self.act1(scale))
        shift = self.shift1(feat, style)
        shift = self.shift2(self.act2(shift))
        residual = feat * scale + shift
        out = feat + residual
        return out

class Synthesis_transformer_SFT(nn.Module):
    def __init__(self, size, style_dim, motion_dim, blur_kernel=[1, 3, 3, 1], channel_multiplier=1):
        super(Synthesis_transformer_SFT, self).__init__()

        self.size = size
        self.style_dim = style_dim
        self.motion_dim = motion_dim

        self.direction = nn.Embedding(motion_dim, style_dim)

        self.channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        self.input = ConstantInput(self.channels[4])
        self.conv1 = StyledConv(self.channels[4], self.channels[4], 3, style_dim, blur_kernel=blur_kernel)
        self.log_size = int(math.log(size, 2))
        self.num_layers = (self.log_size - 2) * 2 + 1

        self.convs = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        self.to_flows = nn.ModuleList()
        self.sft = nn.ModuleList()

        in_channel = self.channels[4]

        for i in range(3, self.log_size + 1):
            out_channel = self.channels[2 ** i]

            self.convs.append(StyledConv(in_channel, out_channel, 3, style_dim, upsample=True,
                                         blur_kernel=blur_kernel))
            self.convs.append(StyledConv(out_channel, out_channel, 3, style_dim, blur_kernel=blur_kernel))
            self.to_rgbs.append(ToRGB(out_channel, style_dim))

            self.to_flows.append(ToFlow(out_channel, style_dim))
            if i > 5:
                self.sft.append(Fuse_sft_block(out_channel, out_channel, style_dim))

            in_channel = out_channel

        self.n_latent = self.log_size * 2 - 2
        self.transformerlayer_0 = TransformerDecoderLayer(d_model=512, nhead=8, dim_feedforward=512)
        self.transformerlayer_1 = TransformerDecoderLayer(d_model=512, nhead=8, dim_feedforward=512)
        self.transformerlayer = TransformerDecoderLayer(d_model=512, nhead=8, dim_feedforward=512)
        self.transformerlayer_t = TransformerDecoderLayer(d_model=512, nhead=8, dim_feedforward=512)
        self.transformerlayer_f = TransformerDecoderLayer(d_model=512, nhead=8, dim_feedforward=512)
        self.latlayer1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.mlp = EqualLinear(512*motion_dim, 512*self.n_latent) 

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True) + y

    def forward(self, was, source, driving, feats):

        bs,_,_,_= was[0].shape
        query_0 = self.direction.weight.unsqueeze(1).repeat(1, bs, 1)
        wa= was[0].flatten(2).permute(2, 0, 1) # HW, B, C
        wa_0 = self.transformerlayer_0(query_0, source, source)
        wa_1 = self.transformerlayer_1(query_0, driving, driving)
        query = self.transformerlayer(wa_0+wa_1, wa, wa)
        wa = was[1].flatten(2).permute(2, 0, 1)
        query = self.transformerlayer_t(query, wa, wa)
        wa = self.latlayer1(was[2]).flatten(2).permute(2, 0, 1)
        query = self.transformerlayer_f(query, wa, wa)
        
        query = query.permute(1,0,2).flatten(1)
        latent = self.mlp(query).view(bs, -1, 512)
        
        out = self.input(latent)
        out = self.conv1(out, latent[:, 0])

        i = 1
        for conv1, conv2, to_rgb, to_flow, feat in zip(self.convs[::2], self.convs[1::2], self.to_rgbs,
                                                       self.to_flows, feats):
            out = conv1(out, latent[:, i])
            out = conv2(out, latent[:, i + 1])

            if out.size(2) == 8:
                out_warp, out, skip_flow = to_flow(out, latent[:, i + 2], feat)
                skip = to_rgb(out_warp)
            else:
                if out.size(2) // 64 > 0 :
                    out_warp, mask, skip_flow = to_flow(out, latent[:, i+2], feat, skip_flow)
                    out_warp = self.sft[int(math.log(out.size(2), 2))-6](out_warp, latent[:, i + 2])
                    out = out_warp + out * (1.0 - mask)
                else:
                    out_warp, out, skip_flow = to_flow(out, latent[:, i + 2], feat, skip_flow)
                
                skip = to_rgb(out_warp, skip)
            i += 2

        img = skip

        return img    

    def forward_relative(self, was, source, driving, feats, driving_start):

        bs,_,_,_= was[0].shape
        query_0 = self.direction.weight.unsqueeze(1).repeat(1, bs, 1)
        wa= was[0].flatten(2).permute(2, 0, 1) # HW, B, C
        wa_0 = self.transformerlayer_0(query_0, source, source)
        wa_01 = self.transformerlayer_1(query_0, source, source)
        wa_start = self.transformerlayer_1(query_0, driving_start, driving_start)
        wa_1 = self.transformerlayer_1(query_0, driving, driving)

        query = self.transformerlayer(wa_0+(wa_1-wa_start)+wa_01, wa, wa)
        wa = was[1].flatten(2).permute(2, 0, 1)
        query = self.transformerlayer_t(query, wa, wa)
        wa = self.latlayer1(was[2]).flatten(2).permute(2, 0, 1)
        query = self.transformerlayer_f(query, wa, wa)

        query = query.permute(1,0,2).flatten(1)
        latent = self.mlp(query).view(bs, -1, 512)

        out = self.input(latent)
        out = self.conv1(out, latent[:, 0])

        i = 1
        for conv1, conv2, to_rgb, to_flow, feat in zip(self.convs[::2], self.convs[1::2], self.to_rgbs,
                                                       self.to_flows, feats):
            out = conv1(out, latent[:, i])
            out = conv2(out, latent[:, i + 1])
            # print(i, out.shape, feat.shape)
            if out.size(2) == 8:
                out_warp, out, skip_flow = to_flow(out, latent[:, i + 2], feat)
                skip = to_rgb(out_warp)
            else:
                if out.size(2) // 64 > 0 :
                    out_warp, mask, skip_flow = to_flow(out, latent[:, i+2], feat, skip_flow)
                    # print(out_warp.shape)
                    out_warp = self.sft[int(math.log(out.size(2), 2))-6](out_warp, latent[:, i + 2])
                    out = out_warp + out * (1.0 - mask)
                else:
                    out_warp, out, skip_flow = to_flow(out, latent[:, i + 2], feat, skip_flow)
                skip = to_rgb(out_warp, skip)
            i += 2

        img = skip

        return img

    def synthesis(self, latent, feats):
        out = self.input(latent)
        out = self.conv1(out, latent[:, 0])

        i = 1
        for conv1, conv2, to_rgb, to_flow, feat in zip(self.convs[::2], self.convs[1::2], self.to_rgbs,
                                                       self.to_flows, feats):
            out = conv1(out, latent[:, i])
            out = conv2(out, latent[:, i + 1])
            # print(i, out.shape, feat.shape)
            if out.size(2) == 8:
                out_warp, out, skip_flow = to_flow(out, latent[:, i + 2], feat)
                skip = to_rgb(out_warp)
            else:
                if out.size(2) // 64 > 0 :
                    out_warp, mask, skip_flow = to_flow(out, latent[:, i+2], feat, skip_flow)
                    # print(out_warp.shape)
                    out_warp = self.sft[int(math.log(out.size(2), 2))-6](out_warp, latent[:, i + 2])
                    out = out_warp + out * (1.0 - mask)
                else:
                    out_warp, out, skip_flow = to_flow(out, latent[:, i + 2], feat, skip_flow)
                skip = to_rgb(out_warp, skip)
            i += 2

        img = skip

        return img
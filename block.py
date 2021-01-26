import torch
import torch.nn as nn
import torch.nn.functional as F
from spectral_normalization import SpectralNorm
from sync_batchnorm.batchnorm import SynchronizedBatchNorm2d


def get_norm_layer(norm_type='batch'):
    def get_out_channel(layer):
        if hasattr(layer, 'out_channels'):
            return getattr(layer, 'out_channels')
        return layer.weight.size(0)

    def add_norm_layer(layer):
        if norm_type == 'batch':
            norm_layer = nn.BatchNorm2d(get_out_channel(layer), affine=True, track_running_stats=True)
        elif norm_type == 'sync_batch':
            norm_layer = SynchronizedBatchNorm2d(get_out_channel(layer), affine=True)
        elif norm_type == 'instance':
            norm_layer = nn.InstanceNorm2d(get_out_channel(layer), affine=False)
        else:
            raise ValueError('normalization layer %s is not recognized' % norm_type)

        return norm_layer

    return add_norm_layer


class ResBlock(nn.Module):
    """Residual Block with instance normalization."""

    def __init__(self, in_channels, out_channels, activation=nn.ReLU(inplace=True), norm_type='batch'):
        super(ResBlock, self).__init__()
        norm_layer = get_norm_layer(norm_type)
        self.main = nn.Sequential(
            norm_layer(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)),
            # nn.BatchNorm2d(out_channels, affine=True, track_running_stats=True),
            activation,
            norm_layer(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False))
            # nn.BatchNorm2d(out_channels, affine=True, track_running_stats=True))
        )

    def forward(self, x):
        return self.main(x) + x


class ConvBlock(nn.Module):
    def __init__(self, dim_in, dim_out, spec_norm=False, LR=0.01, stride=1, up=False, norm_type='batch'):
        super(ConvBlock, self).__init__()

        self.up = up
        if self.up:
            self.up_smaple = nn.UpsamplingBilinear2d(scale_factor=2)
        else:
            self.up_smaple = None

        norm_layer = get_norm_layer(norm_type)

        if spec_norm:
            self.main = nn.Sequential(
                # norm_layer(SpectralNorm(nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=stride, padding=1, bias=False))),
                SpectralNorm(nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=stride, padding=1, bias=False)),
                nn.BatchNorm2d(dim_out, affine=True, track_running_stats=True),
                nn.LeakyReLU(LR, inplace=False),
            )

        else:
            self.main = nn.Sequential(
                nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(dim_out, affine=True, track_running_stats=True),
                nn.LeakyReLU(LR, inplace=False),
            )

    def forward(self, x1, x2=None):
        if self.up_smaple is not None:
            x1 = self.up_smaple(x1)
            # input is CHW
            diffY = x2.size()[2] - x1.size()[2]
            diffX = x2.size()[3] - x1.size()[3]

            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
            # if you have padding issues, see
            # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
            # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
            x = torch.cat([x2, x1], dim=1)
            return self.main(x)
        else:
            return self.main(x1)


"""
class ConvUpBlock(nn.Module):
    def __init__(self, dim_in, dim_out, spec_norm=False, LR=0.01, up=False):
        super(ConvUpBlock, self).__init__()
        self.up = up
        if self.up:
            self.up_smaple = nn.UpsamplingBilinear2d(scale_factor=2)
        else:
            self.up_smaple = None

        if spec_norm:
            self.main = nn.Sequential(
                SpectralNorm(nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False)),
                nn.BatchNorm2d(dim_out, affine=True, track_running_stats=True),
                nn.LeakyReLU(LR, inplace=False),
            )

        else:
            self.main = nn.Sequential(
                nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(dim_out, affine=True, track_running_stats=True),
                nn.LeakyReLU(LR, inplace=False),
            )

    def forward(self, x1, x2):
        if self.up_smaple is not None:
            x1 = self.up_smaple(x1)
            # input is CHW
            diffY = x2.size()[2] - x1.size()[2]
            diffX = x2.size()[3] - x1.size()[3]

            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])
            # if you have padding issues, see
            # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
            # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
            x = torch.cat([x2, x1], dim=1)
            return self.main(x)
        else:
            return self.main(x1)
"""

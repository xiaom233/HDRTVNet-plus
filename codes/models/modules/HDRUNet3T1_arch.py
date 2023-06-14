import functools
import torch
import torch.nn as nn
from torch.nn import init as init
from torch.nn.modules.batchnorm import _BatchNorm
import models.modules.arch_util as arch_util


class HDRUNet3T1(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nf=64, act_type='relu', weighting_network=True):
        super(HDRUNet3T1, self).__init__()

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1)

        self.SFT_layer1 = arch_util.SFTLayer(in_nc=nf//2, out_nc=nf, nf=nf//2)
        self.HR_conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        self.down_conv1 = nn.Conv2d(nf, nf, 3, 2, 1)
        self.down_conv2 = nn.Conv2d(nf, nf, 3, 2, 1)
        self.down_conv3 = nn.Conv2d(nf, nf, 3, 2, 1)

        basic_block = functools.partial(arch_util.ResBlock_with_SFT, nf=nf)
        self.recon_trunk1 = arch_util.make_layer(basic_block, 1)
        self.recon_trunk2 = arch_util.make_layer(basic_block, 1)
        self.recon_trunk3 = arch_util.make_layer(basic_block, 4)
        self.recon_trunk4 = arch_util.make_layer(basic_block, 1)
        self.recon_trunk5 = arch_util.make_layer(basic_block, 1)


        self.up_conv1 = nn.Sequential(nn.Conv2d(nf, nf * 4, 3, 1, 1), nn.PixelShuffle(2))
        self.up_conv2 = nn.Sequential(nn.Conv2d(nf, nf * 4, 3, 1, 1), nn.PixelShuffle(2))
        self.up_conv3 = nn.Sequential(nn.Conv2d(nf, nf * 4, 3, 1, 1), nn.PixelShuffle(2))

        self.SFT_layer2 = arch_util.SFTLayer(in_nc=nf//2, out_nc=nf, nf=nf//2)
        self.HR_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        cond_in_nc = 3
        cond_nf = 64
        self.cond_first = nn.Sequential(nn.Conv2d(cond_in_nc, cond_nf, 3, 1, 1), nn.LeakyReLU(0.1, True),
                                        nn.Conv2d(cond_nf, cond_nf, 1), nn.LeakyReLU(0.1, True),
                                        nn.Conv2d(cond_nf, cond_nf, 1), nn.LeakyReLU(0.1, True))
        self.CondNet1 = nn.Sequential(nn.Conv2d(cond_nf, cond_nf, 1), nn.LeakyReLU(0.1, True),
                                      nn.Conv2d(cond_nf, cond_nf, 1), nn.LeakyReLU(0.1, True),
                                      nn.Conv2d(cond_nf, nf//2, 1))
        self.CondNet2 = nn.Sequential(nn.Conv2d(cond_nf, cond_nf, 3, 2, 1), nn.LeakyReLU(0.1, True),
                                      nn.Conv2d(cond_nf, cond_nf, 1), nn.LeakyReLU(0.1, True),
                                      nn.Conv2d(cond_nf, nf//2, 1))
        self.CondNet3 = nn.Sequential(nn.Conv2d(cond_nf, cond_nf, 3, 2, 1), nn.LeakyReLU(0.1, True),
                                      nn.Conv2d(cond_nf, cond_nf, 3, 2, 1), nn.LeakyReLU(0.1, True),
                                      nn.Conv2d(cond_nf, nf//2, 1))
        self.CondNet4 = nn.Sequential(nn.Conv2d(cond_nf, cond_nf, 3, 2, 1), nn.LeakyReLU(0.1, True),
                                      nn.Conv2d(cond_nf, cond_nf, 3, 2, 1), nn.LeakyReLU(0.1, True),
                                      nn.Conv2d(cond_nf, nf//2, 3, 2, 1))

        self.weighting_network = weighting_network
        if weighting_network:
            self.mask_est = nn.Sequential(nn.Conv2d(in_nc, nf, 3, 1, 1),
                                          nn.ReLU(inplace=True),
                                          nn.Conv2d(nf, nf, 3, 1, 1),
                                          nn.ReLU(inplace=True),
                                          nn.Conv2d(nf, nf, 1),
                                          nn.ReLU(inplace=True),
                                          nn.Conv2d(nf, out_nc, 1),
                                          )

        # activation function
        if act_type == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif act_type == 'leakyrelu':
            self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        # x[0]: img; x[1]: cond
        if self.weighting_network:
            mask = self.mask_est(x[0])
            mask_out = mask * x[0]
        else:
            mask_out = x[0] # long skip connection

        cond = self.cond_first(x[1])
        cond1 = self.CondNet1(cond)
        cond2 = self.CondNet2(cond)
        cond3 = self.CondNet3(cond)
        cond4 = self.CondNet4(cond)

        fea0 = self.act(self.conv_first(x[0]))

        fea0 = self.SFT_layer1((fea0, cond1))
        fea0 = self.act(self.HR_conv1(fea0))

        fea1 = self.act(self.down_conv1(fea0))
        fea1, _ = self.recon_trunk1((fea1, cond2))

        fea2 = self.act(self.down_conv2(fea1))
        fea2, _ = self.recon_trunk2((fea2, cond3))

        fea3 = self.act(self.down_conv3(fea2))
        out, _ = self.recon_trunk3((fea3, cond4))

        out = out + fea3

        out = self.act(self.up_conv1(out)) + fea2
        out, _ = self.recon_trunk4((out, cond3))

        out = self.act(self.up_conv2(out)) + fea1
        out, _ = self.recon_trunk5((out, cond2))

        out = self.act(self.up_conv3(out)) + fea0
        out = self.SFT_layer2((out, cond1))

        out = self.act(self.HR_conv2(out))

        out = self.conv_last(out)
        out = mask_out + out
        return out, x[0]


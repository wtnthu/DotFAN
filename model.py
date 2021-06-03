import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.modules.utils import _pair
from torch.nn.modules import conv, Linear
class ResidualBlock(nn.Module):
    """Residual Block."""
    def __init__(self, dim_in, dim_out,Norm):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            Norm(dim_out, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            Norm(dim_out, affine=True))

    def forward(self, x):
        return x + self.main(x)
latant_number=256
label_size=7
class Generator(nn.Module):
    """Generator. Encoder-Decoder Architecture."""
    def __init__(self, conv_dim=64, c_dim=5, repeat_num=6,input_channel=3,Norm=nn.InstanceNorm2d):
        super(Generator, self).__init__()

        layers = []
        layers.append(nn.Conv2d(input_channel, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(Norm(conv_dim, affine=True))
        layers.append(nn.ReLU(inplace=True))

        # Down-Sampling
        curr_dim = conv_dim
        for i in range(4):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(Norm(curr_dim*2, affine=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2
            for j in range(2):
                layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim,Norm=Norm))

        layers.append(nn.AvgPool2d(label_size,label_size))
        self.enc = nn.Sequential(*layers)

        dec_fc_layer=[]
        dec_fc_layer.append(nn.Linear(latant_number+c_dim,latant_number*10))
        dec_fc_layer.append(nn.Linear(latant_number*10, latant_number * label_size * label_size))
        self.dec_fc=nn.Sequential(*dec_fc_layer)

        '''
        # Bottleneck
        #for i in range(repeat_num//2):
        #    layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))
        #self.latent_dis = latent_dis()
        # curr_dim=curr_dim+c_dim
        '''
        layers_dec=[]


        # Up-Sampling
        for i in range(4):
            for j in range(2):
                layers_dec.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim,Norm=Norm))
            layers_dec.append(nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False))
            layers_dec.append(Norm(curr_dim//2, affine=True))
            layers_dec.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2

        layers_dec.append(nn.Conv2d(curr_dim, input_channel, kernel_size=7, stride=1, padding=3, bias=False))
        layers_dec.append(nn.Tanh())
        self.dec = nn.Sequential(*layers_dec)

    def forward(self, x, c,id_latent,use_feature=[]):
        if len(use_feature)==0:
            latent=self.enc(x)
            latent = latent.view(-1, latant_number)
        else:
            latent = use_feature[0]

        if len(id_latent)==0:
            out = torch.cat([latent, c], dim=1)
        else:
            out = torch.cat([latent, c, id_latent[0]], dim=1)

        out=self.dec_fc(out)
        out=out.view(-1,latant_number,label_size,label_size)

        output = self.dec(out)
        return output,latent

class Discriminator(nn.Module):
    """Discriminator. PatchGAN."""
    def __init__(self, image_size=128, conv_dim=64, c_dim=5, repeat_num=6,input_channel=3):
        super(Discriminator, self).__init__()

        layers = []
        layers.append(nn.Conv2d(input_channel, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01, inplace=True))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01, inplace=True))
            curr_dim = curr_dim * 2

        k_size = int(image_size / np.power(2, repeat_num))
        self.main = nn.Sequential(*layers)
        self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(curr_dim, c_dim, kernel_size=k_size, bias=False)

    def forward(self, x):
        h = self.main(x)
        out_real = self.conv1(h)
        out_aux = self.conv2(h)
        return out_real.squeeze(), out_aux.squeeze()

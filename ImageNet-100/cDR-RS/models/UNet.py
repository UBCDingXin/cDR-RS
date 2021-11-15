'''
Modified based on https://github.com/milesial/Pytorch-UNet

For 3x32x32 input
'''


import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, mid_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()

        if not mid_channels:
            mid_channels = in_channels // 2

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, mid_channels)

    def forward(self, x1, x2=None):
        if x2 is None:
            x1 = self.up(x1)
            return self.conv(x1)
        else:
            x = torch.cat([x2, x1], dim=1)
            x = self.up(x)
            return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.Tanh()
        )
    def forward(self, x):
        return self.conv(x)




class unet_encoder(nn.Module):
    def __init__(self, input_channels=3):
        super(unet_encoder, self).__init__()
        self.inc = DoubleConv(input_channels, 128)
        self.down1 = Down(in_channels=128, out_channels=128, mid_channels=128) #32-->16
        self.down2 = Down(in_channels=128, out_channels=32, mid_channels=128) #16-->8
        self.down3 = Down(in_channels=32, out_channels=64, mid_channels=128) #8-->4

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        return x3, x4



class unet_decoder(nn.Module):
    def __init__(self, output_channels=3):
        super(unet_decoder, self).__init__()
        self.output_channels = output_channels
        self.up1 = Up(64, 64, mid_channels=128) #4-->8
        self.up2 = Up(96, 128, mid_channels=128) #8-->16
        self.up3 = Up(128, 256, mid_channels=128) #16-->32
        self.outc = OutConv(256, output_channels)

    def forward(self, o1, o2):
        x = self.up1(o2)
        x = self.up2(x,o1)
        x = self.up3(x)
        
        x = self.outc(x)
        return x


if __name__ == "__main__":
    def get_parameter_number(net):
        total_num = sum(p.numel() for p in net.parameters())
        trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
        return {'Total': total_num, 'Trainable': trainable_num}

    encoder = unet_encoder().cuda()
    x = torch.randn((5,3,32,32)).cuda()
    o1,o2=encoder(x)
    print(o1.size())
    print(o2.size())

    decoder = unet_decoder().cuda()
    x_hat = decoder(o1,o2)
    print(x_hat.size())

    print(get_parameter_number(encoder))
    print(get_parameter_number(decoder))
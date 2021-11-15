'''
Borrowed from https://github.com/sangwoomo/GOLD
'''

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_norm(use_sn):
    if use_sn:  # spectral normalization
        return nn.utils.spectral_norm
    else:  # identity mapping
        return lambda x: x


# https://github.com/znxlwm/pytorch-generative-model-collections/blob/master/utils.py
def weights_init(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()


# https://github.com/gitlimlab/ACGAN-PyTorch/blob/master/utils.py
def weights_init_3channel(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def onehot(y, class_num):
    eye = torch.eye(class_num).type_as(y)  # ny x ny
    onehot = eye[y.view(-1)].float()  # B -> B x ny
    return onehot


# https://github.com/gitlimlab/ACGAN-PyTorch/blob/master/network.py
class ACGAN_Generator(nn.Module):
    def __init__(self, nz=128, ny=100, nc=3):
        super(ACGAN_Generator, self).__init__()
        self.class_num = ny
        self.fc = nn.Linear(nz + ny, 384)
        self.tconv = nn.Sequential(
            # tconv1
            nn.ConvTranspose2d(384, 192, 4, 1, 0, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(True),
            # tconv2
            nn.ConvTranspose2d(192, 96, 4, 2, 1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(True),
            # tconv3
            nn.ConvTranspose2d(96, 48, 4, 2, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(True),
            # tconv4
            nn.ConvTranspose2d(48, nc, 4, 2, 1, bias=False),
            nn.Tanh(),
        )
        weights_init_3channel(self)

    def forward(self, x, y):
        y = onehot(y, self.class_num)  # B -> B x ny
        x = torch.cat([x, y], dim=1)  # B x (nz + ny)
        x = self.fc(x)
        x = x.view(-1, 384, 1, 1)
        x = self.tconv(x)
        return x


# https://github.com/gitlimlab/ACGAN-PyTorch/blob/master/network.py
class ACGAN_Discriminator(nn.Module):
    def __init__(self, ny=100, nc=3, use_sn=True):
        super(ACGAN_Discriminator, self).__init__()
        norm = get_norm(use_sn)
        self.conv = nn.Sequential(
            # conv1
            norm(nn.Conv2d(nc, 16, 3, 2, 1, bias=False)),  # use spectral norm
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
            # conv2
            norm(nn.Conv2d(16, 32, 3, 1, 1, bias=False)),  # use spectral norm
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
            # conv3
            norm(nn.Conv2d(32, 64, 3, 2, 1, bias=False)),  # use spectral norm
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
            # conv4
            norm(nn.Conv2d(64, 128, 3, 1, 1, bias=False)),  # use spectral norm
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
            # conv5
            norm(nn.Conv2d(128, 256, 3, 2, 1, bias=False)),  # use spectral norm
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
            # conv6
            norm(nn.Conv2d(256, 512, 3, 1, 1, bias=False)),  # use spectral norm
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
        )
        self.out_d = nn.Linear(4 * 4 * 512, 1)
        self.out_c = nn.Linear(4 * 4 * 512, ny)
        weights_init_3channel(self)

    def forward(self, x, y=None, get_feature=False, act_sigmoid=False):
        features = self.conv(x)
        features = features.view(-1, 4*4*512)

        if get_feature:
            return features
        else:
            if act_sigmoid:
                return F.sigmoid(self.out_d(features)), self.out_c(features)
            else:
                return self.out_d(features), self.out_c(features)



if __name__ == "__main__":
    netG = ACGAN_Generator(nz=128, ny=100).cuda()
    netD = ACGAN_Discriminator(ny=100).cuda()

    N=4
    z = torch.randn(N, 128).type(torch.float).cuda()
    y = torch.randint(high=100, size=(N,)).type(torch.long).cuda()
    print(y.shape)
    x = netG(z,y)
    d, y_hat = netD(x,y, act_sigmoid=True)
    print(x.size())
    print(d.size())
    print(y_hat.size())

    def get_parameter_number(net):
        total_num = sum(p.numel() for p in net.parameters())
        trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
        return {'Total': total_num, 'Trainable': trainable_num}
    print(get_parameter_number(netG))
    print(get_parameter_number(netD))
'''

Density Ration Approximation via Multilayer Perceptron

Multilayer Perceptron : trained to model density ratio in a feature space; based on "Rectified Linear Units Improve Restricted Boltzmann Machines"

Its input is the output of a pretrained Deep CNN, say ResNet-34

'''

import torch
import torch.nn as nn


IMG_SIZE=64


class ConditionalNorm2d(nn.Module):
    def __init__(self, num_features, dim_cond, dim_group=None):
        super().__init__()
        self.num_features = num_features
        # self.norm = nn.BatchNorm2d(num_features, affine=False)
        self.norm = nn.GroupNorm(dim_group, num_features, affine=False)

        self.embed_gamma = nn.Linear(dim_cond, num_features, bias=False)
        self.embed_beta = nn.Linear(dim_cond, num_features, bias=False)

    def forward(self, x, y):
        out = self.norm(x)

        gamma = self.embed_gamma(y).view(-1, self.num_features, 1, 1)
        beta = self.embed_beta(y).view(-1, self.num_features, 1, 1)
        out = out + out*gamma + beta

        return out


class cDR_CNN_2(nn.Module):
    def __init__(self, img_size=IMG_SIZE, dim_cond = 128):
        super(cDR_CNN_2, self).__init__()
        self.img_size = img_size
        self.dim_cond = dim_cond

        self.conv1 = nn.Conv2d(3, 128, kernel_size=4, stride=2, padding=1) # 64x64x3 --> 32x32
        self.norm1 = ConditionalNorm2d(128, dim_cond, dim_group=8)

        self.conv2 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1) # 32x32 --> 16x16
        self.norm2 = ConditionalNorm2d(256, dim_cond, dim_group=8)

        self.conv3 = nn.Conv2d(256, 256, kernel_size=4, stride=2, padding=1) # 16x16 --> 8x8
        self.norm3 = ConditionalNorm2d(256, dim_cond, dim_group=16)

        self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1) # 8x8 --> 4x4
        self.norm4 = ConditionalNorm2d(512, dim_cond, dim_group=16)

        self.conv5 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=0) #  4x4 --> 2x2
        self.norm5 = ConditionalNorm2d(512, dim_cond, dim_group=16)
        
        self.relu = nn.ReLU()

        self.final = nn.Sequential(
            nn.Linear(2048, 256),
            nn.GroupNorm(8, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.ReLU()
        )
    
    def forward(self, x, y):
        x = x.view(x.size(0), 3, self.img_size, self.img_size)
        y = y.view(y.size(0), -1)

        ## layer 1
        out = self.conv1(x)
        out = self.norm1(out, y)
        out = self.relu(out)

        ## layer 2
        out = self.conv2(out)
        out = self.norm2(out, y)
        out = self.relu(out)

        ## layer 3
        out = self.conv3(out)
        out = self.norm3(out, y)
        out = self.relu(out)

        ## layer 4
        out = self.conv4(out)
        out = self.norm4(out, y)
        out = self.relu(out)

        ## layer 5
        out = self.conv5(out)
        out = self.norm5(out, y)
        out = self.relu(out)

        ##final
        out = out.view(out.size(0),-1)
        out = self.final(out)

        return out


if __name__ == "__main__":
    init_in_dim = 2
    net = cDR_CNN_2(img_size=64, dim_cond=128).cuda()
    x = torch.randn((10,64**2*3)).cuda()
    labels = torch.randn((10, 128)).cuda()
    out = net(x, labels)
    print(out.size())

    def get_parameter_number(net):
        total_num = sum(p.numel() for p in net.parameters())
        trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
        return {'Total': total_num, 'Trainable': trainable_num}
    print(get_parameter_number(net))

'''

Density Ration Approximation via Multilayer Perceptron

Multilayer Perceptron : trained to model density ratio in a feature space; based on "Rectified Linear Units Improve Restricted Boltzmann Machines"

Its input is the output of a pretrained Deep CNN, say ResNet-34

'''

import torch
import torch.nn as nn


IMG_SIZE=128


class ConditionalNorm2d(nn.Module):
    def __init__(self, num_features, num_classes, dim_group=None):
        super().__init__()
        self.num_features = num_features
        # self.norm = nn.BatchNorm2d(num_features, affine=False)
        self.norm = nn.GroupNorm(dim_group, num_features, affine=False)

        self.embed_gamma = nn.Embedding(num_classes, num_features)
        self.embed_beta = nn.Embedding(num_classes, num_features)

    def forward(self, x, y):
        out = self.norm(x)

        gamma = self.embed_gamma(y).view(-1, self.num_features, 1, 1)
        beta = self.embed_beta(y).view(-1, self.num_features, 1, 1)
        out = out + out*gamma + beta

        return out


class cDR_CNN(nn.Module):
    def __init__(self, img_size=IMG_SIZE, num_classes = 100):
        super(cDR_CNN, self).__init__()
        self.img_size = img_size
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1) # 128x128x3 --> 64x64
        self.norm1 = ConditionalNorm2d(64, num_classes, dim_group=8)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1) # 64x64 --> 32x32
        self.norm2 = ConditionalNorm2d(128, num_classes, dim_group=8)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1) # 32x32 --> 16x16
        self.norm3 = ConditionalNorm2d(256, num_classes, dim_group=16)

        self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1) # 16x16 --> 8x8
        self.norm4 = ConditionalNorm2d(512, num_classes, dim_group=16)

        self.conv5 = nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1) #  8x8 --> 4x4
        self.norm5 = ConditionalNorm2d(1024, num_classes, dim_group=16)
        
        self.relu = nn.ReLU()
        self.avgpool = nn.AvgPool2d(4,4)

        self.final = nn.Sequential(
            nn.Linear(1024, 256),
            nn.GroupNorm(8, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.ReLU()
        )
    
    def forward(self, x, y):
        x = x.view(x.size(0), 3, self.img_size, self.img_size)

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
        out = self.avgpool(out)
        out = out.view(out.size(0),-1)
        out = self.final(out)

        return out


if __name__ == "__main__":
    num_classes=100
    net = cDR_CNN(img_size=128, num_classes=num_classes).cuda()
    x = torch.randn((10,128**2*3)).cuda()
    labels = torch.LongTensor(10).random_(0, num_classes).cuda()
    out = net(x, labels)
    print(out.size())

    def get_parameter_number(net):
        total_num = sum(p.numel() for p in net.parameters())
        trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
        return {'Total': total_num, 'Trainable': trainable_num}
    print(get_parameter_number(net))

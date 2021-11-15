import torch
import torch.nn as nn
from torchvision.utils import save_image
import numpy as np
import os
import timeit

from utils import *
from opts import gen_synth_data_opts


''' Settings '''
args = gen_synth_data_opts()

lr_d = 1e-4
gan_net = args.gan_net
epochs = args.keep_training_epochs
dim_gan = args.gan_dim_g
loss_type = "hinge"


def finetune_netD(trainloader, netG, netD):
    netG = netG.cuda()
    netD = netD.cuda()
    netG = netG.eval()

    optimizerD = torch.optim.Adam(netD.parameters(), lr=lr_d, betas=(0.5, 0.999))

    start_time = timeit.default_timer()
    for epoch in range(epochs):

        for batch_idx, (batch_real_images, batch_real_labels) in enumerate(trainloader):
            netD.train()

            batch_size_curr = batch_real_images.shape[0]


            ''' get real images '''
            batch_real_images = batch_real_images.type(torch.float).cuda()
            batch_real_labels = batch_real_labels.type(torch.long).cuda()


            ''' get some fake images '''
            with torch.no_grad():
                z = torch.randn(batch_size_curr, dim_gan, dtype=torch.float).cuda()
                if gan_net in ["BigGANdeep", "BigGAN"]:
                    batch_fake_images = netG(z, netG.module.shared(batch_real_labels))
                else:
                    batch_fake_images = netG(z, batch_real_labels)
                batch_fake_images = batch_fake_images.detach()

            ''' Train Discriminator: maximize log(D(x)) + log(1 - D(G(z)))'''

            # Measure discriminator's ability to classify real from generated samples
            real_dis_out = netD(batch_real_images, batch_real_labels)
            fake_dis_out = netD(batch_fake_images.detach(), batch_real_labels.detach())
                
            
            if loss_type == "vanilla":
                real_dis_out = torch.nn.Sigmoid()(real_dis_out)
                fake_dis_out = torch.nn.Sigmoid()(fake_dis_out)
                d_loss_real = - torch.log(real_dis_out+1e-20)
                d_loss_fake = - torch.log(1-fake_dis_out+1e-20)
            elif loss_type == "hinge":
                d_loss_real = torch.nn.ReLU()(1.0 - real_dis_out)
                d_loss_fake = torch.nn.ReLU()(1.0 + fake_dis_out)
            d_loss = (d_loss_real + d_loss_fake).mean()

            optimizerD.zero_grad()
            d_loss.backward()
            optimizerD.step()

            if (batch_idx+1)%20 == 0:
                print ("Finetune netD: [epoch %d/%d] [batch_idx %d/%d] [D loss: %.4f] [D out real:%.4f] [D out fake:%.4f] [Time: %.4f]" % (epoch, epochs, batch_idx+1, len(trainloader), d_loss.item(), real_dis_out.mean().item(),fake_dis_out.mean().item(), timeit.default_timer()-start_time))

    #end for niter
    netD = netD.cpu()
    netG = netG.cpu()

    return netD

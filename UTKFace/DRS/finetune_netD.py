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
niters = args.keep_training_niters
dim_gan = args.gan_dim_g
dim_embed = args.dim_embed
batch_size = args.keep_training_batchsize
loss_type = "vanilla"
max_label = args.max_label

## normalize images
def normalize_images(batch_images):
    batch_images = batch_images/255.0
    batch_images = (batch_images - 0.5)/0.5
    return batch_images

def finetune_netD(images, labels, netG, netD, net_y2h):

    '''
    unnormalized images and labels
    '''

    netG = netG.cuda()
    netD = netD.cuda()
    netG = netG.eval()

    optimizerD = torch.optim.Adam(netD.parameters(), lr=lr_d, betas=(0.5, 0.999))

    labels = labels/max_label
    assert images.max()>1.0 and images.max()<=255.0
    assert labels.max()<=1.0 and labels.min()>=0
    unique_labels = np.sort(np.array(list(set(labels))))

    start_time = timeit.default_timer()
    for niter in range(niters):

        ''' generate target labels '''
        batch_target_labels = np.random.choice(unique_labels, size=batch_size, replace=True)
        batch_unique_labels, batch_unique_label_counts = np.unique(batch_target_labels, return_counts=True)

        batch_real_indx = []
        for j in range(len(batch_unique_labels)):
            indx_j = np.where(labels==batch_unique_labels[j])[0]
            indx_j = np.random.choice(indx_j, size=batch_unique_label_counts[j])
            batch_real_indx.append(indx_j)
        batch_real_indx = np.concatenate(batch_real_indx)
        batch_real_indx = batch_real_indx.reshape(-1)

        ''' get some real images for training '''
        batch_train_images = images[batch_real_indx]
        batch_train_images = normalize_images(batch_train_images) ## normalize real images
        batch_train_images = torch.from_numpy(batch_train_images).type(torch.float).cuda()
        assert batch_train_images.max().item()<=1.0
        batch_train_labels = labels[batch_real_indx]
        batch_train_labels = torch.from_numpy(batch_train_labels).type(torch.float).view(-1,1).cuda()
        batch_train_labels = net_y2h(batch_train_labels)


        ''' Train Discriminator: maximize log(D(x)) + log(1 - D(G(z)))'''
        # Sample noise and labels as generator input
        z = torch.randn(batch_size, dim_gan, dtype=torch.float).cuda()

        #generate fake images
        batch_fake_images = netG(z, batch_train_labels)

        # Measure discriminator's ability to classify real from generated samples
        real_dis_out = netD(batch_train_images, batch_train_labels)
        fake_dis_out = netD(batch_fake_images.detach(), batch_train_labels.detach())
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

        if (niter+1)%20 == 0:
            print ("Finetune netD: [Iter %d/%d] [D loss: %.4f] [D out real:%.4f] [D out fake:%.4f] [Time: %.4f]" % (niter+1, niters, d_loss.item(), real_dis_out.mean().item(),fake_dis_out.mean().item(), timeit.default_timer()-start_time))
    #end for niter
    netD = netD.cpu()
    netG = netG.cpu()

    return netD

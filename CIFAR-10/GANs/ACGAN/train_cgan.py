import torch
import torch.nn as nn
from torchvision.utils import save_image
import numpy as np
import os
import timeit

from utils import IMGs_dataset, SimpleProgressBar
from opts import parse_opts
from DiffAugment_pytorch import DiffAugment
from eval_metrics import compute_IS
from models import *


##################################################################
''' Settings '''
args = parse_opts()

# some parameters in opts
niters = args.niters
resume_niters = args.resume_niter
dim_gan = args.dim_gan
lr_g = args.lr_g
lr_d = args.lr_d
save_freq = args.save_freq
num_classes = args.num_classes
num_D_steps = args.num_D_steps
lambda_aux_fake = args.lambda_aux_fake #default: 0.1

visualize_freq = args.visualize_freq

num_channels = args.num_channels
img_size = args.img_size

use_DiffAugment = args.gan_DiffAugment
policy = args.gan_DiffAugment_policy



##################################################################
## compute IS in training
path_torch_home = os.path.join(args.root_path, 'torch_cache')
os.makedirs(path_torch_home, exist_ok=True)
os.environ['TORCH_HOME'] = path_torch_home

comp_IS_in_train = args.comp_IS_in_train
comp_IS_freq = args.comp_IS_freq
comp_IS_batch_size = 100

if comp_IS_in_train:
    if args.inception_from_scratch:
        #load pre-trained InceptionV3 (pretrained on CIFAR-10)
        PreNetIS = Inception3(num_classes=args.num_classes, aux_logits=True, transform_input=False)
        checkpoint_PreNet = torch.load(args.eval_ckpt_path)
        PreNetIS = nn.DataParallel(PreNetIS)
        PreNetIS.load_state_dict(checkpoint_PreNet['net_state_dict'])
    else:
        PreNetIS = inception_v3(pretrained=True, transform_input=True)
        PreNetIS = nn.DataParallel(PreNetIS)


def fn_generate_fake_samples(pretrained_netG, nfake_per_class=100, num_classes=num_classes, batch_size=args.samp_batch_size):
    
    fake_labels = []
    for i in range(num_classes):
        fake_labels_i = (i*torch.ones(nfake_per_class)).type(torch.long)
        fake_labels.append(fake_labels_i)
    fake_labels = torch.cat(fake_labels)

    fake_images = []
    pretrained_netG = pretrained_netG.cuda()
    pretrained_netG.eval()
    with torch.no_grad():
        tmp = 0
        while tmp < int(nfake_per_class*num_classes):
            z = torch.randn(batch_size, dim_gan, dtype=torch.float).cuda()
            batch_fake_labels = (fake_labels[tmp:(tmp+batch_size)]).type(torch.long).cuda()
            batch_fake_images = pretrained_netG(z, batch_fake_labels)
            fake_images.append(batch_fake_images.cpu())
            tmp += batch_size

    fake_images = torch.cat(fake_images, dim=0)

    fake_images = fake_images.numpy()
    fake_labels = fake_labels.numpy()

    return fake_images[0:int(nfake_per_class*num_classes)], fake_labels[0:int(nfake_per_class*num_classes)]



##################################################################
class GANLoss(nn.Module):
    def __init__(self, target_real_label=1.0, target_fake_label=0.0, reduction='mean'):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.loss = nn.BCEWithLogitsLoss(reduction=reduction)

    def get_target_tensor(self, prediction, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        target_tensor = self.get_target_tensor(prediction, target_is_real)
        loss = self.loss(prediction, target_tensor)
        return loss

def train_cgan(trainloader, netG, netD, save_images_folder, save_models_folder = None):

    netG = netG.cuda()
    netD = netD.cuda()

    optimizerG = torch.optim.Adam(netG.parameters(), lr=lr_g, betas=(0.5, 0.999))
    optimizerD = torch.optim.Adam(netD.parameters(), lr=lr_d, betas=(0.5, 0.999))

    # initialize criterion
    criterionGAN = GANLoss(reduction='none').cuda()
    criterionCE = nn.CrossEntropyLoss(reduction='none').cuda()


    if save_models_folder is not None and resume_niters>0:
        save_file = save_models_folder + "/ACGAN_nDsteps_{}_checkpoint_intrain/ACGAN_checkpoint_niters_{}_lambda_{}.pth".format(num_D_steps, resume_niters, lambda_aux_fake)
        checkpoint = torch.load(save_file)
        netG.load_state_dict(checkpoint['netG_state_dict'])
        netD.load_state_dict(checkpoint['netD_state_dict'])
        optimizerG.load_state_dict(checkpoint['optimizerG_state_dict'])
        optimizerD.load_state_dict(checkpoint['optimizerD_state_dict'])
        torch.set_rng_state(checkpoint['rng_state'])
    #end if

    ## fixed noise and labels for image generation during training.
    n_row=10
    n_col=10
    unique_labels = np.arange(num_classes)
    selected_labels = np.zeros(n_row)
    indx_step_size = len(unique_labels)//n_row
    for i in range(n_row):
        indx = i*indx_step_size
        selected_labels[i] = unique_labels[indx]
    y_fixed = np.zeros(n_row*n_col)
    for i in range(n_row):
        curr_label = selected_labels[i]
        for j in range(n_col):
            y_fixed[i*n_col+j] = curr_label
    y_fixed = torch.from_numpy(y_fixed).type(torch.long).cuda()
    z_fixed = torch.randn(len(y_fixed), dim_gan, dtype=torch.float).cuda()

    batch_idx = 0
    dataloader_iter = iter(trainloader)

    start_time = timeit.default_timer()
    for niter in range(resume_niters, niters):

        if batch_idx+1 == len(trainloader):
            dataloader_iter = iter(trainloader)
            batch_idx = 0

        '''

        Train Generator: maximize log(D(G(z)))

        '''

        netG.train()

        # get training images
        _, batch_train_labels = dataloader_iter.next()
        batch_size = batch_train_labels.shape[0]
        batch_train_labels = batch_train_labels.type(torch.long).cuda()
        batch_idx+=1

        # Sample noise and labels as generator input
        z = torch.randn(batch_size, dim_gan, dtype=torch.float).cuda()

        #generate fake images
        batch_fake_images = netG(z, batch_train_labels)

        # Loss measures generator's ability to fool the discriminator
        if use_DiffAugment:
            dis_out, dis_y_hat = netD(DiffAugment(batch_fake_images, policy=policy), batch_train_labels)
        else:
            dis_out, dis_y_hat = netD(batch_fake_images, batch_train_labels)

        g_loss_adv = torch.mean(criterionGAN(dis_out, True)) #use True or False??? https://github.com/sangwoomo/GOLD/blob/0feb18a78a2c56001dbbd8eabf807bbb409d25db/train/train_acgan_full.py#L77
        g_loss_aux = torch.mean(criterionCE(dis_y_hat, batch_train_labels))
        
        g_loss = g_loss_adv + g_loss_aux

        optimizerG.zero_grad()
        g_loss.backward()
        optimizerG.step()

        '''

        Train Discriminator: maximize log(D(x)) + log(1 - D(G(z)))

        '''

        for _ in range(num_D_steps):

            if batch_idx+1 == len(trainloader):
                dataloader_iter = iter(trainloader)
                batch_idx = 0

            # get training images
            batch_train_images, batch_train_labels = dataloader_iter.next()
            assert batch_size == batch_train_images.shape[0]
            batch_train_images = batch_train_images.type(torch.float).cuda()
            batch_train_labels = batch_train_labels.type(torch.long).cuda()
            batch_idx+=1

            # Measure discriminator's ability to classify real from generated samples
            if use_DiffAugment:
                real_dis_out, real_y_hat = netD(DiffAugment(batch_train_images, policy=policy), batch_train_labels)
                fake_dis_out, fake_y_hat = netD(DiffAugment(batch_fake_images.detach(), policy=policy), batch_train_labels)
            else:
                real_dis_out, real_y_hat = netD(batch_train_images, batch_train_labels)
                fake_dis_out, fake_y_hat = netD(batch_fake_images.detach(), batch_train_labels)

            ## real loss
            d_loss_adv_real = torch.mean(criterionGAN(real_dis_out, True))
            d_loss_aux_real = torch.mean(criterionCE(real_y_hat, batch_train_labels))

            ## fake loss
            d_loss_adv_fake = torch.mean(criterionGAN(fake_dis_out, False))
            d_loss_aux_fake = torch.mean(criterionCE(fake_y_hat, batch_train_labels)) 

            d_loss = d_loss_adv_real + d_loss_aux_real + d_loss_adv_fake + d_loss_aux_fake * lambda_aux_fake

            optimizerD.zero_grad()
            d_loss.backward()
            optimizerD.step()

        
        if (niter+1)%20 == 0:
            print ("\n ACGAN: [Iter %d/%d] [D loss: %.4f] [G loss: %.4f] [D out real:%.4f] [yh real: %.4f] [D out fake:%.4f] [yh fake: %.4f] [Time: %.4f]" % (niter+1, niters, d_loss.item(), g_loss.item(), real_dis_out.mean().item(), d_loss_aux_real.item(), fake_dis_out.mean().item(), d_loss_aux_fake.item(), timeit.default_timer()-start_time))

        if (niter+1) % visualize_freq == 0:
            netG.eval()
            with torch.no_grad():
                gen_imgs = netG(z_fixed, y_fixed)
                gen_imgs = gen_imgs.detach()
            save_image(gen_imgs.data, save_images_folder +'/{}.png'.format(niter+1), nrow=n_col, normalize=True)
        
        if comp_IS_in_train and (niter+1)%comp_IS_freq==0:
            fake_images_eval,_ = fn_generate_fake_samples(pretrained_netG=netG)
            IS_mean, IS_std = compute_IS(PreNetIS, fake_images_eval, batch_size = comp_IS_batch_size, splits=10, resize=(299,299), verbose=False)
            print('\n ACGAN: [Iter: {}/{}] [IS mean/std: {}/{}]'.format(niter+1, niters, IS_mean, IS_std))

        if save_models_folder is not None and ((niter+1) % save_freq == 0 or (niter+1) == niters):
            save_file = save_models_folder + "/ACGAN_nDsteps_{}_checkpoint_intrain/ACGAN_checkpoint_niters_{}_lambda_{}.pth".format(num_D_steps, niter+1, lambda_aux_fake)
            os.makedirs(os.path.dirname(save_file), exist_ok=True)
            torch.save({
                    'netG_state_dict': netG.state_dict(),
                    'netD_state_dict': netD.state_dict(),
                    'optimizerG_state_dict': optimizerG.state_dict(),
                    'optimizerD_state_dict': optimizerD.state_dict(),
                    'rng_state': torch.get_rng_state()
            }, save_file)
    #end for niter

    return netG, netD
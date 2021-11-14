'''

Functions for Training Class-conditional Density-ratio model

'''

import torch
import torch.nn as nn
import numpy as np
import os
import timeit
import gc

from utils import *
from opts import gen_synth_data_opts

''' Settings '''
args = gen_synth_data_opts()

# some parameters in the opts
dim_gan = args.gan_dim_g
dre_net_name = args.dre_net
dre_epochs = args.dre_epochs
dre_lr_base = args.dre_lr_base
dre_lr_decay_factor = args.dre_lr_decay_factor
dre_lr_decay_epochs = (args.dre_lr_decay_epochs).split("_")
dre_lr_decay_epochs = [int(epoch) for epoch in dre_lr_decay_epochs]
dre_lambda = args.dre_lambda
dre_resume_epoch = args.dre_resume_epoch
dre_batch_size = args.dre_batch_size
dre_optimizer = args.dre_optimizer
dre_save_freq = args.dre_save_freq


## normalize images
def normalize_images(batch_images):
    batch_images = batch_images/255.0
    batch_images = (batch_images - 0.5)/0.5
    return batch_images

# training function
def train_cdre(kappa, unique_labels, train_images, train_labels, dre_net, dre_precnn_net, netG, net_y2h, net_predict=None, reg_niters=10, path_to_ckpt=None):
    ##data; train_images are unnormalized, train_labels are normalized
    ## unique_labels: normalized unique labels
    assert train_images.max()>1.0 and train_images.max()<=255.0
    assert train_labels.max()<=1.0 and train_labels.min()>=0
    assert 0<=kappa<=1

    ''' learning rate decay '''
    def adjust_learning_rate(optimizer, epoch):
        lr = dre_lr_base
        num_decays = len(dre_lr_decay_epochs)
        for decay_i in range(num_decays):
            if epoch >= dre_lr_decay_epochs[decay_i]:
                lr = lr * dre_lr_decay_factor
            #end if epoch
        #end for decay_i
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    #end def adjust lr


    # nets
    dre_precnn_net = dre_precnn_net.cuda()
    netG = netG.cuda()
    net_y2h = net_y2h.cuda()
    dre_net = dre_net.cuda()
    
    dre_precnn_net.eval()
    netG.eval()
    net_y2h.eval()
    
    if net_predict is not None: #the predicting branch of the sparse AE
        net_predict = net_predict.cuda()
        net_predict.eval()


    # define optimizer
    if dre_optimizer=="SGD":
        optimizer = torch.optim.SGD(dre_net.parameters(), lr = dre_lr_base, momentum= 0.9, weight_decay=1e-4)
    else:
        optimizer = torch.optim.Adam(dre_net.parameters(), lr = dre_lr_base, betas=(0.5, 0.999), weight_decay=1e-4)

    if path_to_ckpt is not None and dre_resume_epoch>0:
        print("Loading ckpt to resume training dre_net >>>")
        ckpt_fullpath = path_to_ckpt + "/cDRE_{}_checkpoint_epoch_{}.pth".format(dre_net_name, dre_resume_epoch)
        checkpoint = torch.load(ckpt_fullpath)
        dre_net.load_state_dict(checkpoint['net_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        #load d_loss and g_loss
        logfile_fullpath = path_to_ckpt + "/cDRE_{}_train_loss_epoch_{}.npz".format(dre_net_name, dre_resume_epoch)
        if os.path.isfile(logfile_fullpath):
            avg_train_loss = list(np.load(logfile_fullpath))
        else:
            avg_train_loss = []
    else:
        avg_train_loss = []


    start_time = timeit.default_timer()
    for epoch in range(dre_resume_epoch, dre_epochs):

        adjust_learning_rate(optimizer, epoch)

        train_loss = 0

        for batch_idx in range(len(train_labels)//dre_batch_size):
            dre_net.train()

            #################################################
            ''' generate target labels '''
            batch_target_labels = np.random.choice(unique_labels, size=dre_batch_size, replace=True)
            batch_unique_labels, batch_unique_label_counts = np.unique(batch_target_labels, return_counts=True)

            batch_real_indx = []
            for j in range(len(batch_unique_labels)):
                indx_j = np.where(train_labels==batch_unique_labels[j])[0]
                indx_j = np.random.choice(indx_j, size=batch_unique_label_counts[j])
                batch_real_indx.append(indx_j)
            batch_real_indx = np.concatenate(batch_real_indx)
            batch_real_indx = batch_real_indx.reshape(-1)

            #################################################
            ''' density ratios of real images '''
            ## get some real images for training
            batch_real_images = train_images[batch_real_indx]
            batch_real_images = normalize_images(batch_real_images) ## normalize real images
            batch_real_images = torch.from_numpy(batch_real_images).type(torch.float).cuda()
            assert batch_real_images.max().item()<=1.0
            batch_real_labels = train_labels[batch_real_indx]
            batch_real_labels = torch.from_numpy(batch_real_labels).type(torch.float).view(-1,1).cuda()

            
            #################################################
            ''' density ratios of fake images '''
            ## generate fake labels first
            if kappa>1e-30:
                batch_fake_labels = np.zeros(dre_batch_size)
                vicinity_start = torch.zeros(dre_batch_size).cuda()
                vicinity_end = torch.zeros(dre_batch_size).cuda()
                for j in range(dre_batch_size):
                    # start_j = max(0, batch_real_labels[j].item()-kappa)
                    # end_j = min(1, batch_real_labels[j].item()+kappa)
                    start_j = batch_real_labels[j].item()-kappa
                    end_j = batch_real_labels[j].item()+kappa
                    assert batch_real_labels[j].item()>=start_j and batch_real_labels[j].item()<=end_j
                    batch_fake_labels[j] = np.random.uniform(low=start_j, high=end_j, size=1)
                    vicinity_start[j] = start_j
                    vicinity_end[j] = end_j
                batch_fake_labels = torch.from_numpy(batch_fake_labels).type(torch.float).view(-1,1).cuda()
                ## then, generate fake images
                ## drop fake images with predicted labels not in the vicinity
                with torch.no_grad():
                    z = torch.randn(dre_batch_size, dim_gan, dtype=torch.float).cuda()
                    batch_fake_images = netG(z, net_y2h(batch_fake_labels))
                    batch_fake_images = batch_fake_images.detach()
                    batch_fake_labels_pred = net_predict(dre_precnn_net(batch_fake_images))
                    indx_drop_1 = batch_fake_labels_pred.view(-1)<vicinity_start
                    indx_drop_2 = batch_fake_labels_pred.view(-1)>vicinity_end
                    indx_drop = torch.cat((indx_drop_1.view(-1,1), indx_drop_2.view(-1,1)), dim=1)
                    indx_drop = torch.any(indx_drop, 1)
                    
                    ## regenerate fake images whose labels are not in the vicinity; at most niter_tmp rounds
                    niter_tmp = 0
                    while indx_drop.sum().item()>0 and niter_tmp<=reg_niters:
                        batch_size_tmp = indx_drop.sum().item()
                        batch_fake_labels_tmp = batch_fake_labels[indx_drop]
                        assert len(batch_fake_labels_tmp)==batch_size_tmp
                        ##update corresponding fake images
                        z = torch.randn(batch_size_tmp, dim_gan, dtype=torch.float).cuda()
                        batch_fake_images_tmp = netG(z, net_y2h(batch_fake_labels_tmp))
                        batch_fake_images[indx_drop] = batch_fake_images_tmp 
                        batch_fake_labels_pred[indx_drop] = net_predict(dre_precnn_net(batch_fake_images_tmp))
                        ##update indices of dropped images
                        indx_drop_1 = batch_fake_labels_pred.view(-1)<vicinity_start
                        indx_drop_2 = batch_fake_labels_pred.view(-1)>vicinity_end
                        indx_drop = torch.cat((indx_drop_1.view(-1,1), indx_drop_2.view(-1,1)),dim=1)
                        indx_drop = torch.any(indx_drop, 1)
                        niter_tmp+=1
                        # print(niter_tmp, indx_drop.sum().item(), dre_batch_size)
                    ###end while
                    
                    indx_keep = (batch_fake_labels_pred.view(-1)>=vicinity_start)*(batch_fake_labels_pred.view(-1)<=vicinity_end)
                    assert indx_keep.sum().item()>0
                    batch_fake_images = batch_fake_images[indx_keep]
                    batch_real_images = batch_real_images[indx_keep] ##if do not do subsampling for real images too, the cDRE training does not converge
                    batch_real_labels = batch_real_labels[indx_keep] ##note that, here is batch_real_labels not batch_fake_labels!!!!
            else:
                with torch.no_grad():
                    z = torch.randn(dre_batch_size, dim_gan, dtype=torch.float).cuda()
                    batch_fake_images = netG(z, net_y2h(batch_real_labels))
                    batch_fake_images = batch_fake_images.detach()

            ## extract features from real and fake images
            with torch.no_grad():
                batch_features_real = dre_precnn_net(batch_real_images)
                batch_features_real = batch_features_real.detach()
                batch_features_fake = dre_precnn_net(batch_fake_images)
                batch_features_fake = batch_features_fake.detach()
            del batch_real_images, batch_fake_images; gc.collect()    

            ## density ratios for real images
            DR_real = dre_net(batch_features_real, net_y2h(batch_real_labels))
            ## density ratios for fake images
            DR_fake = dre_net(batch_features_fake, net_y2h(batch_real_labels)) ##Please note that use batch_real_labels here !!!!
                

            
            #################################################
            #Softplus loss
            softplus_fn = torch.nn.Softplus(beta=1,threshold=20)
            sigmoid_fn = torch.nn.Sigmoid()
            SP_div = torch.mean(sigmoid_fn(DR_fake) * DR_fake) - torch.mean(softplus_fn(DR_fake)) - torch.mean(sigmoid_fn(DR_real))
            penalty = dre_lambda * (torch.mean(DR_fake) - 1)**2
            loss = SP_div + penalty

            #backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.cpu().item()

            print("cDRE+{}+lambda{}: [step {}/{}] [epoch {}/{}] [train loss {:.5f}] [fake batch {}/{}] [Time {:.4f}]".format(dre_net_name, dre_lambda, batch_idx+1, len(train_labels)//dre_batch_size, epoch+1, dre_epochs, train_loss/(batch_idx+1), len(batch_features_fake), dre_batch_size, timeit.default_timer()-start_time))

            # #################################################
            # ### debugging
            # dre_net.eval()
            # with torch.no_grad():
            #     DR_real2 = dre_net(batch_features_real, net_y2h(batch_real_labels))
            #     DR_fake2 = dre_net(batch_features_fake, net_y2h(batch_fake_labels))
            #     print("[Iter {}/{}], [epoch {}/{}], Debug (train):{:.4f}/{:.4f}".format(batch_idx, len(train_labels)//dre_batch_size, epoch+1, dre_epochs, DR_real.mean(),DR_fake.mean()))
            #     print("[Iter {}/{}], [epoch {}/{}], Debug (eval):{:.4f}/{:.4f}".format(batch_idx, len(train_labels)//dre_batch_size, epoch+1, dre_epochs, DR_real2.mean(),DR_fake2.mean()))
        # end for batch_idx

        # print("cDRE+{}+lambda{}: [epoch {}/{}] [train loss {}] [Time {}]".format(dre_net_name, dre_lambda, epoch+1, dre_epochs, train_loss/(batch_idx+1), timeit.default_timer()-start_time))

        avg_train_loss.append(train_loss/(batch_idx+1))

        # save checkpoint
        if path_to_ckpt is not None and ((epoch+1) % dre_save_freq == 0 or (epoch+1)==dre_epochs):
            ckpt_fullpath = path_to_ckpt + "/cDRE_{}_checkpoint_epoch_{}.pth".format(dre_net_name, epoch+1)
            torch.save({
                    'epoch': epoch,
                    'net_state_dict': dre_net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
            }, ckpt_fullpath)

            # save loss
            logfile_fullpath = path_to_ckpt + "/cDRE_{}_train_loss_epoch_{}.npz".format(dre_net_name, epoch+1)
            np.savez(logfile_fullpath, np.array(avg_train_loss))

    #end for epoch

    #back to memory
    dre_precnn_net = dre_precnn_net.cpu()
    netG = netG.cpu()
    net_y2h = net_y2h.cpu()
    dre_net = dre_net.cpu()
    if net_predict is not None:
        net_predict = net_predict.cpu()

    return dre_net, avg_train_loss
#end for def

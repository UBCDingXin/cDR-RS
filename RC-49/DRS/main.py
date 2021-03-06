print("\n ===================================================================================================")


#----------------------------------------
import argparse
import os
import timeit
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.nn import functional as F
import random
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('Agg')
from torch import autograd
from torchvision.utils import save_image
from tqdm import tqdm, trange
import gc
from itertools import groupby
import multiprocessing
import h5py
import pickle
import copy
import shutil
import math

#----------------------------------------
from opts import gen_synth_data_opts
from utils import *
from models import *
from finetune_netD import finetune_netD
from eval_metrics import cal_FID, cal_labelscore



#######################################################################################
'''                                   Settings                                      '''
#######################################################################################
args = gen_synth_data_opts()
print(args)


if args.subsampling:
    subsampling_method = "DRS_keepTrain_{}".format(args.keep_training)
else:
    subsampling_method = "None"


if "hard" in args.gan_ckpt_path:
    gan_name = "HVDL+ILI"
else:
    gan_name = "SVDL+ILI"


#-------------------------------
# seeds
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
cudnn.benchmark = False
np.random.seed(args.seed)

#-------------------------------
# output folders
precnn_models_directory = os.path.join(args.root_path, 'output/precnn_models')
os.makedirs(precnn_models_directory, exist_ok=True)

output_directory = os.path.join(args.root_path, 'output/{}'.format(gan_name))
os.makedirs(output_directory, exist_ok=True)

save_models_folder = os.path.join(output_directory, 'saved_models')
os.makedirs(save_models_folder, exist_ok=True)

save_traincurves_folder = os.path.join(output_directory, 'training_curves')
os.makedirs(save_traincurves_folder, exist_ok=True)

save_evalresults_folder = os.path.join(output_directory, 'eval_results')
os.makedirs(save_evalresults_folder, exist_ok=True)

dump_fake_images_folder = os.path.join(output_directory, 'dump_fake')
os.makedirs(dump_fake_images_folder, exist_ok=True)


#######################################################################################
'''                                  Load Data                                      '''
#######################################################################################
data_filename = args.data_path + '/RC-49_{}x{}.h5'.format(args.img_size, args.img_size)
hf = h5py.File(data_filename, 'r')
labels_all = hf['labels'][:]
labels_all = labels_all.astype(float)
images_all = hf['images'][:]
indx_train = hf['indx_train'][:]
hf.close()
print("\n RC-49 dataset shape: {}x{}x{}x{}".format(images_all.shape[0], images_all.shape[1], images_all.shape[2], images_all.shape[3]))

# data split
if args.data_split == "train":
    images_train = images_all[indx_train]
    labels_train_raw = labels_all[indx_train]
else:
    images_train = copy.deepcopy(images_all)
    labels_train_raw = copy.deepcopy(labels_all)

# only take images with label in (q1, q2)
q1 = args.min_label
q2 = args.max_label
indx = np.where((labels_train_raw>q1)*(labels_train_raw<q2)==True)[0]
labels_train_raw = labels_train_raw[indx]
images_train = images_train[indx]
assert len(labels_train_raw)==len(images_train)

indx = np.where((labels_all>q1)*(labels_all<q2)==True)[0]
labels_all = labels_all[indx]
images_all = images_all[indx]
assert len(labels_all)==len(images_all)


### show some real  images
if args.show_real_imgs:
    unique_labels_show = np.array(sorted(list(set(labels_all))))
    indx_show = np.arange(0, len(unique_labels_show), len(unique_labels_show)//9)
    unique_labels_show = unique_labels_show[indx_show]
    nrow = len(unique_labels_show); ncol = 1
    sel_labels_indx = []
    for i in range(nrow):
        curr_label = unique_labels_show[i]
        indx_curr_label = np.where(labels_all==curr_label)[0]
        np.random.shuffle(indx_curr_label)
        indx_curr_label = indx_curr_label[0:ncol]
        sel_labels_indx.extend(list(indx_curr_label))
    sel_labels_indx = np.array(sel_labels_indx)
    images_show = images_all[sel_labels_indx]
    print(images_show.mean())
    images_show = (images_show/255.0-0.5)/0.5
    images_show = torch.from_numpy(images_show)
    save_image(images_show.data, save_evalresults_folder +'/real_images_grid_{}x{}.png'.format(nrow, ncol), nrow=ncol, normalize=True)

# for each angle, take no more than args.max_num_img_per_label images
image_num_threshold = args.max_num_img_per_label
print("\n Original set has {} images; For each angle, take no more than {} images>>>".format(len(images_train), image_num_threshold))
unique_labels_tmp = np.sort(np.array(list(set(labels_train_raw))))
for i in tqdm(range(len(unique_labels_tmp))):
    indx_i = np.where(labels_train_raw == unique_labels_tmp[i])[0]
    if len(indx_i)>image_num_threshold:
        np.random.shuffle(indx_i)
        indx_i = indx_i[0:image_num_threshold]
    if i == 0:
        sel_indx = indx_i
    else:
        sel_indx = np.concatenate((sel_indx, indx_i))
images_train = images_train[sel_indx]
labels_train_raw = labels_train_raw[sel_indx]
print("{} images left and there are {} unique labels".format(len(images_train), len(set(labels_train_raw))))


# normalize labels
print("\n Range of unnormalized labels: ({},{})".format(np.min(labels_train_raw), np.max(labels_train_raw)))
labels_train = labels_train_raw / args.max_label #normalize to [0,1]
print("\n Range of normalized labels: ({},{})".format(np.min(labels_train), np.max(labels_train)))

unique_labels_train_norm = np.sort(np.array(list(set(labels_train)))) ##sorted unique labels




#######################################################################################
'''                  Load pre-trained GAN to Memory (not GPU)                       '''
#######################################################################################

print("\n Loading pre-trained embedding network, i.e., net_y2h ...")
net_y2h = model_y2h(dim_embed=args.dim_embed)
checkpoint = torch.load(args.embed_ckpt_path)
net_y2h.load_state_dict(checkpoint['net_state_dict'])
net_y2h = nn.DataParallel(net_y2h)

print("\n Loading pre-trained CcGAN ...")
checkpoint = torch.load(args.gan_ckpt_path)
netG = CcGAN_Generator(nz=args.gan_dim_g, dim_embed=args.dim_embed)
netG = nn.DataParallel(netG)
netG.load_state_dict(checkpoint['netG_state_dict'])

netD = CcGAN_Discriminator(dim_embed=args.dim_embed)
netD = nn.DataParallel(netD)
netD.load_state_dict(checkpoint['netD_state_dict'])


if args.keep_training:
    print("\n Finetuning netD for DRS...")
    filename_netD_keeptrain = os.path.join(save_models_folder, 'ckpt_netD_keepTrain_niters_{}.pth'.format(args.keep_training_niters))
    if not os.path.isfile(filename_netD_keeptrain):
        netD = finetune_netD(images=images_train, labels=labels_train_raw, netG=netG, netD=netD, net_y2h=net_y2h)
        # save model
        torch.save({
        'netD_state_dict': netD.state_dict(),
        }, filename_netD_keeptrain)
    else:
        checkpoint_netD_net = torch.load(filename_netD_keeptrain)
        netD.load_state_dict(checkpoint_netD_net['netD_state_dict'])
    ##end if not


def fn_sampleGAN_given_label(nfake, given_label, netG=netG, net_y2h=net_y2h, batch_size = 500, to_numpy=True, denorm=True, verbose=False):
    ''' given_label: normalized label '''
    netG = netG.cuda()
    net_y2h = net_y2h.cuda()
    netG.eval()
    net_y2h.eval()

    if batch_size>nfake:
        batch_size = nfake
    
    fake_images = []
    with torch.no_grad():
        if verbose:
            pb = SimpleProgressBar()
        n_img_got = 0
        while n_img_got < nfake:
            y = np.ones(batch_size) * given_label ##normalized labels
            y = torch.from_numpy(y).type(torch.float).view(-1,1).cuda()
            z = torch.randn(batch_size, args.gan_dim_g, dtype=torch.float).cuda()
            batch_fake_images = netG(z, net_y2h(y))
            if denorm:
                batch_fake_images = (batch_fake_images*0.5+0.5)*255.0
                batch_fake_images = batch_fake_images.type(torch.uint8)
            fake_images.append(batch_fake_images.cpu())
            n_img_got += len(batch_fake_images)
            if verbose:
                pb.update(min(float(n_img_got)/nfake, 1)*100)
    fake_images = torch.cat(fake_images, dim=0)
    fake_labels = (torch.ones(nfake) * given_label).type(torch.float) #use assigned label

    if to_numpy:
        fake_images = fake_images.numpy()
        fake_labels = fake_labels.numpy()

    netG = netG.cpu()
    net_y2h = net_y2h.cpu()

    return fake_images[0:nfake], fake_labels[0:nfake]


def comp_cond_density_ratio(imgs, labels, batch_size = args.samp_batch_size, netD=netD, net_y2h=net_y2h):
    #imgs: a torch tensor; unnormalized
    #labels: normalized labels

    assert imgs.max()>=1.0 and imgs.max()<=255.0
    assert labels.max()<=1.0 and labels.min()>=0

    netD = netD.cuda()
    net_y2h = net_y2h.cuda()
    netD.eval()
    net_y2h.eval()

    n_imgs = len(imgs)
    if batch_size>n_imgs:
        batch_size = n_imgs

    ##make sure the last iteration has enough samples
    imgs = torch.cat((imgs, imgs[0:batch_size]), dim=0)
    labels = torch.cat((labels, labels[0:batch_size]), dim=0)

    density_ratios = []
    # print("\n Begin computing density ratio for images >>")
    with torch.no_grad():
        n_imgs_got = 0
        while n_imgs_got < n_imgs:
            batch_images = imgs[n_imgs_got:(n_imgs_got+batch_size)]
            batch_images = (batch_images/255.0-0.5)/0.5 ##normalized images
            batch_labels = labels[n_imgs_got:(n_imgs_got+batch_size)]
            batch_images = batch_images.type(torch.float).cuda()
            batch_labels = batch_labels.type(torch.float).view(-1,1).cuda()
            batch_labels = net_y2h(batch_labels)
            disc_probs = netD(batch_images, batch_labels, use_sigmoid=True).cpu().detach().numpy()
            disc_probs = np.clip(disc_probs.astype(float), 1e-14, 1 - 1e-14)
            density_ratios.append(np.divide(disc_probs, 1-disc_probs))
            n_imgs_got += batch_size
        ### while n_imgs_got
    density_ratios = np.concatenate(density_ratios)
    density_ratios = density_ratios[0:n_imgs]
    return density_ratios


def fn_enhancedSampler_given_label(nfake, given_label, batch_size=args.samp_batch_size, verbose=True):
    ''' given_label: normalized '''

    ## Burn-in Stage
    n_burnin = args.samp_burnin_size

    burnin_imgs, burnin_labels = fn_sampleGAN_given_label(nfake=n_burnin, given_label=given_label, batch_size = batch_size, to_numpy=False, denorm=True, verbose=False)
    burnin_densityratios = comp_cond_density_ratio(burnin_imgs, burnin_labels)
    print((burnin_densityratios.min(),np.median(burnin_densityratios),burnin_densityratios.max()))
    M_bar = np.max(burnin_densityratios)
    del burnin_imgs, burnin_densityratios; gc.collect()
    
    ## Rejection sampling
    enhanced_imgs = []
    if verbose:
        pb = SimpleProgressBar()
        # pbar = tqdm(total=nfake)
    num_imgs = 0
    while num_imgs < nfake:

        batch_imgs, batch_labels = fn_sampleGAN_given_label(nfake=batch_size, given_label=given_label, batch_size = batch_size, to_numpy=False, denorm=True, verbose=False)
        batch_ratios = comp_cond_density_ratio(batch_imgs, batch_labels)
        M_bar = np.max([M_bar, np.max(batch_ratios)])
        
        #threshold
        epsilon_tmp = 1e-8;
        D_tilde_M = np.log(M_bar)
        batch_F = np.log(batch_ratios) - D_tilde_M - np.log(1-np.exp(np.log(batch_ratios)-D_tilde_M-epsilon_tmp))
        gamma_tmp = np.percentile(batch_F, 80) #80 percentile of each batch; follow DRS's setting
        batch_F_hat = batch_F - gamma_tmp
        batch_p = 1/(1+np.exp(-batch_F_hat))

        batch_psi = np.random.uniform(size=batch_size).reshape(-1,1)
        indx_accept = np.where(batch_psi<=batch_p)[0]
        if len(indx_accept)>0:
            enhanced_imgs.append(batch_imgs[indx_accept])
        num_imgs+=len(indx_accept)
        del batch_imgs, batch_ratios; gc.collect()
        if verbose:
            pb.update(np.min([float(num_imgs)*100/nfake,100]))
            # pbar.update(len(indx_accept))
    # pbar.close()
    enhanced_imgs = np.concatenate(enhanced_imgs, axis=0)
    enhanced_imgs = enhanced_imgs[0:nfake]
    return enhanced_imgs, given_label*np.ones(nfake)



###############################################################################
'''                               Evaluation                                '''
###############################################################################
if args.eval or args.samp_dump_fake_data:
    
    #####################
    ## evaluate models on givenlabels
    if args.eval_mode == 1: #Mode 1: eval on unique labels used for GAN training
        eval_labels = np.sort(np.array(list(set(labels_train_raw)))) #not normalized
    elif args.eval_mode in [2, 3]: #Mode 2 and 3: eval on all unique labels in the dataset
        eval_labels = np.sort(np.array(list(set(labels_all)))) #not normalized
    else: #Mode 4: eval on a interval [min_label, max_label] with num_eval_labels labels
        eval_labels = np.linspace(np.min(labels_all), np.max(labels_all), args.num_eval_labels) #not normalized

    eval_labels_norm = eval_labels/args.max_label #normalized
    num_eval_labels = len(eval_labels_norm)
    print("\n There are {} distinct eval labels: MIN={}, MAX={}".format(len(list(set(eval_labels))), eval_labels_norm.min(), eval_labels_norm.max()))

    #####################
    # generate nfake images: ONE h5 file
    print("\n Start sampling {} fake images per label from GAN >>>".format(args.samp_nfake_per_label))

    dump_fake_images_filename = os.path.join(dump_fake_images_folder, 'fake_images_CcGAN_subsampling_{}_nfakePerLabel_{}_nfake_{}_seed_{}.h5'.format(subsampling_method, args.samp_nfake_per_label, int(num_eval_labels*args.samp_nfake_per_label), args.seed))
    print('\n '+dump_fake_images_filename)

    if not os.path.isfile(dump_fake_images_filename):
        print('\n Start generating fake data...')
        fake_images = []
        fake_labels = []
        start = timeit.default_timer()
        for i in range(num_eval_labels):
            print("\n Start generating {} fake images for label {}/{}.".format(args.samp_nfake_per_label, i+1, num_eval_labels))
            if args.subsampling:
                fake_images_i, fake_labels_i = fn_enhancedSampler_given_label(nfake=args.samp_nfake_per_label, given_label=eval_labels_norm[i], batch_size=args.samp_batch_size, verbose=True)
            else:
                fake_images_i, fake_labels_i = fn_sampleGAN_given_label(nfake=args.samp_nfake_per_label, given_label=eval_labels_norm[i], batch_size=args.samp_batch_size, to_numpy=True, denorm=True, verbose=True)
            assert fake_images_i.max()>1 and fake_images_i.max()<=255.0
            assert fake_labels_i.min()>=0 and fake_labels_i.max()<=1
            fake_images.append(fake_images_i)
            fake_labels.append(fake_labels_i.reshape(-1))
            print("\r Generated {} fake images for label {}/{}. Time elapses: {}".format(args.samp_nfake_per_label, i+1, num_eval_labels, timeit.default_timer()-start))
        ##end for i
        fake_images = np.concatenate(fake_images, axis=0)
        fake_labels = np.concatenate(fake_labels, axis=0)
        del fake_images_i, fake_labels_i; gc.collect()
        assert fake_labels.min()>=0 and fake_labels.max()<=1
        print('\n End generating fake data!')
        fake_labels = (fake_labels*args.max_label).astype(float) #denormalize fake labels

        if args.samp_dump_fake_data:
            with h5py.File(dump_fake_images_filename, "w") as f:
                f.create_dataset('fake_images', data = fake_images, dtype='uint8', compression="gzip", compression_opts=6)
                f.create_dataset('fake_labels', data = fake_labels, dtype='float')
    else:
        print('\n Start loading generated fake data...')
        with h5py.File(dump_fake_images_filename, "r") as f:
            fake_images = f['fake_images'][:]
            fake_labels = f['fake_labels'][:]
    assert len(fake_images) == len(fake_labels)
    fake_labels = np.round_(fake_labels, decimals=4) #denormalize fake labels
    print("\r Unnormalized fake labels' range: MIN={}, MAX={}".format(fake_labels.min(), fake_labels.max()))
    print("\r There are {} distinct fake labels..".format(len(list(set(fake_labels)))))
    print("End sampling!")
    print("\n We got {} fake images.".format(len(fake_images)))


    ## dump fake images for evaluation: NIQE
    if args.dump_fake_for_NIQE:
        print("\n Dumping fake images for NIQE...")
        path_to_imgs_for_NIQE = os.path.join(dump_fake_images_folder, 'fake_images_for_NIQE_CcGAN_subsampling_{}_NfakePerLabel_{}_seed_{}'.format(subsampling_method, args.samp_nfake_per_label, args.seed))
        os.makedirs(path_to_imgs_for_NIQE, exist_ok=True)
        for i in tqdm(range(len(fake_images))):
            # label_i = round(fake_labels[i]*args.max_label, 4)
            label_i = round(fake_labels[i], 4)
            filename_i = path_to_imgs_for_NIQE + "/{}_{}.png".format(i, label_i)
            os.makedirs(os.path.dirname(filename_i), exist_ok=True)
            image_i = fake_images[i]
            image_i_pil = Image.fromarray(image_i.transpose(1,2,0))
            image_i_pil.save(filename_i)
        #end for i
        sys.exit()
    

    #####################
    # Evaluation
    if args.eval:
        print("\n Start evaluation...")

        PreNetFID = encoder_eval(dim_bottleneck=512).cuda()
        PreNetFID = nn.DataParallel(PreNetFID)
        checkpoint_PreNet = torch.load(args.eval_ckpt_path_FID)
        PreNetFID.load_state_dict(checkpoint_PreNet['net_encoder_state_dict'])

        # Diversity: entropy of predicted races within each eval center
        PreNetDiversity = ResNet34_class_eval(num_classes=49, ngpu = torch.cuda.device_count()).cuda() #5 races
        checkpoint_PreNet = torch.load(args.eval_ckpt_path_Div)
        PreNetDiversity.load_state_dict(checkpoint_PreNet['net_state_dict'])

        # for LS
        PreNetLS = ResNet34_regre_eval(ngpu = torch.cuda.device_count()).cuda()
        checkpoint_PreNet = torch.load(args.eval_ckpt_path_LS)
        PreNetLS.load_state_dict(checkpoint_PreNet['net_state_dict'])


        if args.eval_mode in [1, 3]:
            real_images = images_train
            real_labels = labels_train_raw #not normalized
        else: #for both mode 2 and 4
            real_images = images_all
            real_labels = labels_all #not normalized
        assert real_labels.max()>1.0

        #######################
        # For each label take nreal_per_label images
        unique_labels_real = np.sort(np.array(list(set(real_labels))))
        indx_subset = []
        for i in range(len(unique_labels_real)):
            label_i = unique_labels_real[i]
            indx_i = np.where(real_labels==label_i)[0]
            np.random.shuffle(indx_i)
            if args.nreal_per_label>1:
                indx_i = indx_i[0:args.nreal_per_label]
            indx_subset.append(indx_i)
        indx_subset = np.concatenate(indx_subset)
        real_images = real_images[indx_subset]
        real_labels = real_labels[indx_subset]

        nfake_all = len(fake_images)
        nreal_all = len(real_images)

        ## normalize labels
        assert fake_labels.max()>1.0
        fake_labels = fake_labels/args.max_label
        assert real_labels.max()>1.0
        real_labels = real_labels/args.max_label

        #####################
        # Evaluate FID within a sliding window with a radius R on the label's range (not normalized range, i.e., [min_label,max_label]). The center of the sliding window locate on [min_label+R,...,max_label-R].
        if args.eval_mode == 1:
            center_start = np.min(labels_train_raw)+args.FID_radius ##bug???
            center_stop = np.max(labels_train_raw)-args.FID_radius
        else:
            center_start = np.min(labels_all)+args.FID_radius
            center_stop = np.max(labels_all)-args.FID_radius

        if args.FID_num_centers<=0 and args.FID_radius==0: #completely overlap
            centers_loc = eval_labels #not normalized
        elif args.FID_num_centers>0:
            centers_loc = np.linspace(center_start, center_stop, args.FID_num_centers) #not normalized
        else:
            print("\n Error.")
            
        FID_over_centers = np.zeros(len(centers_loc))
        entropies_over_centers = np.zeros(len(centers_loc)) # entropy at each center
        labelscores_over_centers = np.zeros(len(centers_loc)) #label score at each center
        num_realimgs_over_centers = np.zeros(len(centers_loc))
        start = timeit.default_timer()
        for i in range(len(centers_loc)):
            center = centers_loc[i]
            interval_start = (center - args.FID_radius)/args.max_label
            interval_stop = (center + args.FID_radius)/args.max_label
           
            ## real images for i-th label
            indx_real = np.where((real_labels>=interval_start)*(real_labels<=interval_stop)==True)[0]
            np.random.shuffle(indx_real)
            real_images_curr = real_images[indx_real]
            real_images_curr = (real_images_curr/255.0-0.5)/0.5
            num_realimgs_over_centers[i] = len(real_images_curr)
            assert len(real_images_curr)>1
            
            ## fake images for i-th label
            indx_fake = np.where((fake_labels>=interval_start)*(fake_labels<=interval_stop)==True)[0]
            np.random.shuffle(indx_fake)
            fake_images_curr = fake_images[indx_fake]
            fake_images_curr = (fake_images_curr/255.0-0.5)/0.5
            fake_labels_assigned_curr = fake_labels[indx_fake]
            assert len(fake_images_curr)>1

            # FID
            FID_over_centers[i] = cal_FID(PreNetFID, real_images_curr, fake_images_curr, batch_size=args.eval_batch_size, resize = None)

            # Entropy of predicted class labels
            predicted_class_labels = predict_class_labels(PreNetDiversity, fake_images_curr, batch_size=args.eval_batch_size, num_workers=args.num_workers)
            entropies_over_centers[i] = compute_entropy(predicted_class_labels)
            # Label score
            labelscores_over_centers[i], _ = cal_labelscore(PreNetLS, fake_images_curr, fake_labels_assigned_curr, min_label_before_shift=0, max_label_after_shift=args.max_label, batch_size = args.eval_batch_size, resize = None, num_workers=args.num_workers)

            print("\n [{}/{}] Center:{}; Real:{}; Fake:{}; FID:{}; LS:{}; ET:{}. Time elapses: {}".format(i+1, len(centers_loc), center, len(real_images_curr), len(fake_images_curr), FID_over_centers[i], labelscores_over_centers[i], entropies_over_centers[i], timeit.default_timer()-start))
        # end for i
        # average over all centers
        print("\n SFID: {}({}); min/max: {}/{}.".format(np.mean(FID_over_centers), np.std(FID_over_centers), np.min(FID_over_centers), np.max(FID_over_centers)))
        print("\n LS over centers: {}({}); min/max: {}/{}.".format(np.mean(labelscores_over_centers), np.std(labelscores_over_centers), np.min(labelscores_over_centers), np.max(labelscores_over_centers)))
        print("\n entropy over centers: {}({}); min/max: {}/{}.".format(np.mean(entropies_over_centers), np.std(entropies_over_centers), np.min(entropies_over_centers), np.max(entropies_over_centers)))

        # dump FID versus number of samples (for each center) to npy
        dump_fid_ls_entropy_over_centers_filename = os.path.join(save_evalresults_folder, 'fid_ls_entropy_over_centers_subsampling_{}'.format(subsampling_method))
        np.savez(dump_fid_ls_entropy_over_centers_filename, fids=FID_over_centers, labelscores=labelscores_over_centers, entropies=entropies_over_centers, nrealimgs=num_realimgs_over_centers, centers=centers_loc)

        #####################
        # FID: Evaluate FID on all fake images
        indx_shuffle_real = np.arange(nreal_all); np.random.shuffle(indx_shuffle_real)
        indx_shuffle_fake = np.arange(nfake_all); np.random.shuffle(indx_shuffle_fake)
        FID = cal_FID(PreNetFID, real_images[indx_shuffle_real], fake_images[indx_shuffle_fake], batch_size=args.eval_batch_size, resize = None, norm_img = True)
        print("\n FID of {} fake images: {}.".format(nfake_all, FID))

        #####################
        # Overall LS: abs(y_assigned - y_predicted)
        ls_mean_overall, ls_std_overall = cal_labelscore(PreNetLS, fake_images, fake_labels, min_label_before_shift=0, max_label_after_shift=args.max_label, batch_size=args.eval_batch_size, resize = None, norm_img = True, num_workers=args.num_workers)
        print("\n overall LS of {} fake images: {}({}).".format(nfake_all, ls_mean_overall, ls_std_overall))
        
        #####################
        # Dump evaluation results
        eval_results_logging_fullpath = os.path.join(save_evalresults_folder, 'eval_results_subsampling_{}.txt'.format(subsampling_method))
        if not os.path.isfile(eval_results_logging_fullpath):
            eval_results_logging_file = open(eval_results_logging_fullpath, "w")
            eval_results_logging_file.close()
        with open(eval_results_logging_fullpath, 'a') as eval_results_logging_file:
            eval_results_logging_file.write("\n===================================================================================================")
            eval_results_logging_file.write("\n Radius: {}.  \n".format(args.FID_radius))
            print(args, file=eval_results_logging_file)
            eval_results_logging_file.write("\n SFID: {}({}).".format(np.mean(FID_over_centers), np.std(FID_over_centers)))
            eval_results_logging_file.write("\n LS: {}({}).".format(np.mean(labelscores_over_centers), np.std(labelscores_over_centers)))
            eval_results_logging_file.write("\n Overall LS: {}({}).".format(ls_mean_overall, ls_std_overall))
            eval_results_logging_file.write("\n Diversity: {}({}).".format(np.mean(entropies_over_centers), np.std(entropies_over_centers)))

print("\n ===================================================================================================")

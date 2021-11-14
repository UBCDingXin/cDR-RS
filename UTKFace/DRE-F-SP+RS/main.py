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


#----------------------------------------
from opts import gen_synth_data_opts
from utils import *
from models import *
from train_dre import train_dre
from train_sparseAE import train_sparseAE
from eval_metrics import cal_FID, cal_labelscore



#######################################################################################
'''                                   Settings                                      '''
#######################################################################################
args = gen_synth_data_opts()
print(args)

if args.subsampling:
    subsampling_method = "DRE-F-SP_presae_sparsity_{:.3f}_regre_{:.3f}_DR_{}_lambda_{:.3f}".format(args.dre_presae_lambda_sparsity, args.dre_presae_lambda_regression, args.dre_net, args.dre_lambda)
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
data_filename = args.data_path + '/UTKFace_{}x{}.h5'.format(args.img_size, args.img_size)
hf = h5py.File(data_filename, 'r')
labels = hf['labels'][:]
labels = labels.astype(float)
images = hf['images'][:]
hf.close()


# subset of UTKFace
selected_labels = np.arange(args.min_label, args.max_label+1)
for i in range(len(selected_labels)):
    curr_label = selected_labels[i]
    index_curr_label = np.where(labels==curr_label)[0]
    if i == 0:
        images_subset = images[index_curr_label]
        labels_subset = labels[index_curr_label]
    else:
        images_subset = np.concatenate((images_subset, images[index_curr_label]), axis=0)
        labels_subset = np.concatenate((labels_subset, labels[index_curr_label]))
# for i
images = images_subset
labels = labels_subset
del images_subset, labels_subset; gc.collect()

raw_images = copy.deepcopy(images)
raw_labels = copy.deepcopy(labels)

### show some real  images
if args.show_real_imgs:
    unique_labels_show = sorted(list(set(labels)))
    nrow = len(unique_labels_show); ncol = 10
    images_show = np.zeros((nrow*ncol, images.shape[1], images.shape[2], images.shape[3]))
    for i in range(nrow):
        curr_label = unique_labels_show[i]
        indx_curr_label = np.where(labels==curr_label)[0]
        np.random.shuffle(indx_curr_label)
        indx_curr_label = indx_curr_label[0:ncol]
        for j in range(ncol):
            images_show[i*ncol+j,:,:,:] = images[indx_curr_label[j]]
    print(images_show.shape)
    images_show = (images_show/255.0-0.5)/0.5
    images_show = torch.from_numpy(images_show)
    save_image(images_show.data, save_evalresults_folder +'/real_images_grid_{}x{}.png'.format(nrow, ncol), nrow=ncol, normalize=True)

# for each age, take no more than args.max_num_img_per_label images
image_num_threshold = args.max_num_img_per_label
print("\n Original set has {} images; For each age, take no more than {} images>>>".format(len(images), image_num_threshold))
unique_labels_tmp = np.sort(np.array(list(set(labels))))
for i in tqdm(range(len(unique_labels_tmp))):
    indx_i = np.where(labels == unique_labels_tmp[i])[0]
    if len(indx_i)>image_num_threshold:
        np.random.shuffle(indx_i)
        indx_i = indx_i[0:image_num_threshold]
    if i == 0:
        sel_indx = indx_i
    else:
        sel_indx = np.concatenate((sel_indx, indx_i))
images = images[sel_indx]
labels = labels[sel_indx]
print("{} images left.".format(len(images)))

## replicate minority samples to alleviate the imbalance
max_num_img_per_label_after_replica = np.min([args.max_num_img_per_label_after_replica, args.max_num_img_per_label])
if max_num_img_per_label_after_replica>1:
    unique_labels_replica = np.sort(np.array(list(set(labels))))
    num_labels_replicated = 0
    print("Start replicating monority samples >>>")
    for i in tqdm(range(len(unique_labels_replica))):
        # print((i, num_labels_replicated))
        curr_label = unique_labels_replica[i]
        indx_i = np.where(labels == curr_label)[0]
        if len(indx_i) < max_num_img_per_label_after_replica:
            num_img_less = max_num_img_per_label_after_replica - len(indx_i)
            indx_replica = np.random.choice(indx_i, size = num_img_less, replace=True)
            if num_labels_replicated == 0:
                images_replica = images[indx_replica]
                labels_replica = labels[indx_replica]
            else:
                images_replica = np.concatenate((images_replica, images[indx_replica]), axis=0)
                labels_replica = np.concatenate((labels_replica, labels[indx_replica]))
            num_labels_replicated+=1
    #end for i
    images = np.concatenate((images, images_replica), axis=0)
    labels = np.concatenate((labels, labels_replica))
    print("We replicate {} images and labels \n".format(len(images_replica)))
    del images_replica, labels_replica; gc.collect()


## distinct unnormalized labels
unique_unnorm_labels = (np.sort(np.array(list(set(labels))))).astype(int) ##sorted unique labels
unnorm_label_to_indx = dict() ##unnormalized label to index
for i in range(len(unique_unnorm_labels)):
    unnorm_label_to_indx[unique_unnorm_labels[i]] = int(i)


#function for (de)normalizing labels
def do_label_norm(labels):
    return labels/args.max_label

def do_label_denorm(labels):
    if torch.is_tensor(labels):
        assert labels.min().item()>=0 and labels.max().item()<=1
        return (labels*args.max_label).type(torch.int)
    else:
        assert labels.min()>=0 and labels.max()<=1
        return (labels*args.max_label).astype(int)


# # normalize labels
# print("\n Range of unnormalized labels: ({},{})".format(np.min(labels), np.max(labels)))
# labels /= args.max_label #normalize to [0,1]
# print("\n Range of normalized labels: ({},{})".format(np.min(labels), np.max(labels)))




#######################################################################################
'''                   Load embedding networks and CcGAN                             '''
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

# netD = CcGAN_Discriminator(dim_embed=args.dim_embed)
# netD = nn.DataParallel(netD)
# netD.load_state_dict(checkpoint['netD_state_dict'])


def fn_sampleGAN_given_label(nfake, given_label, netG=netG, net_y2h=net_y2h, batch_size = 500, to_numpy=True, denorm=True, verbose=False):
    ''' given_label: unnormalized label '''

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
            y = do_label_norm(np.ones(batch_size) * given_label) ##normalized labels
            y = torch.from_numpy(y).type(torch.float).view(-1,1).cuda()
            z = torch.randn(batch_size, args.gan_dim_g, dtype=torch.float).cuda()
            batch_fake_images = netG(z, net_y2h(y))
            ##end if vicinal
            if denorm:
                batch_fake_images = (batch_fake_images*0.5+0.5)*255.0
                batch_fake_images = batch_fake_images.type(torch.uint8)
            fake_images.append(batch_fake_images.cpu())
            n_img_got += len(batch_fake_images)
            if verbose:
                pb.update(min(float(n_img_got)/nfake, 1)*100)
    fake_images = torch.cat(fake_images, dim=0)
    fake_labels = (torch.ones(nfake) * given_label).type(torch.int) #use assigned label

    if to_numpy:
        fake_images = fake_images.numpy()
        fake_labels = fake_labels.numpy()

    netG = netG.cpu()
    net_y2h = net_y2h.cpu()

    return fake_images[0:nfake], fake_labels[0:nfake]




#######################################################################################
'''                                  DRE Training                                   '''
#######################################################################################
if args.subsampling:
    ##############################################
    ''' Pre-trained CNN for feature extraction '''
    print("\n -----------------------------------------------------------------------------------------")
    print("\n Pre-trained CNN for feature extraction")
    
    # filename
    filename_presae_ckpt = precnn_models_directory + '/ckpt_PreSAEForDRE_epoch_{}_sparsity_{:.3f}_regre_{:.3f}_seed_{}.pth'.format(args.dre_presae_epochs, args.dre_presae_lambda_sparsity, args.dre_presae_lambda_regression, args.seed)
    print('\n ' + filename_presae_ckpt)

    # training
    if not os.path.isfile(filename_presae_ckpt):

        save_sae_images_InTrain_folder = precnn_models_directory + '/PreSAEForDRE_reconstImages_sparsity_{:.3f}_regre_{:.3f}_InTrain_{}'.format(args.dre_presae_lambda_sparsity, args.dre_presae_lambda_regression, args.seed)
        os.makedirs(save_sae_images_InTrain_folder, exist_ok=True)

        # dataloader
        trainset = IMGs_dataset(images, do_label_norm(labels), normalize=True) ##labels are normalized
        trainloader_sparseAE = torch.utils.data.DataLoader(trainset, batch_size=args.dre_presae_batch_size_train, shuffle=True, num_workers=args.num_workers)

        # initialize net
        dre_presae_encoder_net = encoder_extract(ch=args.dre_presae_ch, dim_bottleneck=args.img_size*args.img_size*args.num_channels)
        dre_presae_decoder_net = decoder_extract(ch=args.dre_presae_ch, dim_bottleneck=args.img_size*args.img_size*args.num_channels)
        dre_presae_predict_net = decoder_predict(dim_bottleneck=args.img_size*args.img_size*args.num_channels)
        dre_presae_encoder_net = nn.DataParallel(dre_presae_encoder_net)
        dre_presae_decoder_net = nn.DataParallel(dre_presae_decoder_net)
        dre_presae_predict_net = nn.DataParallel(dre_presae_predict_net)

        count_parameters(dre_presae_encoder_net)
        count_parameters(dre_presae_decoder_net)
        count_parameters(dre_presae_predict_net)

        print("\n Start training sparseAE model for feature extraction in the DRE >>>")
        dre_presae_encoder_net, dre_presae_decoder_net, dre_presae_predict_net = train_sparseAE(trainloader=trainloader_sparseAE, net_encoder=dre_presae_encoder_net, net_decoder=dre_presae_decoder_net, net_predict=dre_presae_predict_net, save_sae_images_folder=save_sae_images_InTrain_folder, path_to_ckpt=precnn_models_directory)
        # store model
        torch.save({
            'encoder_net_state_dict': dre_presae_encoder_net.state_dict(),
            'predict_net_state_dict': dre_presae_predict_net.state_dict(),
            # 'decoder_net_state_dict': dre_presae_decoder_net.state_dict(),
        }, filename_presae_ckpt)
        print("\n End training CNN.")
    else:
        print("\n Loading pre-trained sparseAE for feature extraction in DRE.")
        dre_presae_encoder_net = encoder_extract(ch=args.dre_presae_ch, dim_bottleneck=args.img_size*args.img_size*args.num_channels)
        dre_presae_encoder_net = nn.DataParallel(dre_presae_encoder_net)
        checkpoint = torch.load(filename_presae_ckpt)
        dre_presae_encoder_net.load_state_dict(checkpoint['encoder_net_state_dict'])
    #end if



    ##############################################
    ''' DRE Training '''
    print("\n -----------------------------------------------------------------------------------------")
    print("\n DRE training")

    dre_net_list = []
    for i in range(len(unique_unnorm_labels)):
        label_i = unique_unnorm_labels[i]
        print("\n Fit DRE-F-SP for Label {}...".format(label_i))

        ### data loader
        indx_label_i = np.where(labels==label_i)[0]
        assert len(indx_label_i)>0
        images_i = images[indx_label_i]
        labels_i = labels[indx_label_i]
        trainset_dre_i = IMGs_dataset(images_i, labels_i, normalize=True) ##labels are unnormalized!!!
        trainloader_dre_i = torch.utils.data.DataLoader(trainset_dre_i, batch_size=args.dre_batch_size, shuffle=True, num_workers=args.num_workers)
        del images_i, labels_i; gc.collect()

        ## dre filename
        drefile_fullpath = save_models_folder + '/ckpt_DRE-F-SP_{}_epochs_{}_lambda_{}_seed_{}_label_{}_of_{}.pth'.format(args.dre_net, args.dre_epochs, args.dre_lambda, args.seed, label_i, args.max_label)
        print('\n' + drefile_fullpath)

        path_to_ckpt_in_train = save_models_folder + '/ckpt_DRE-F-SP_{}_lambda_{}_seed_{}_label_{}_of_{}'.format(args.dre_net, args.dre_lambda, args.seed, label_i, args.max_label)
        os.makedirs(path_to_ckpt_in_train, exist_ok=True)

        dre_loss_file_fullpath = save_traincurves_folder + '/train_loss_DRE-F-SP_{}_epochs_{}_lambda_{}_seed_{}_label_{}_of_{}.png'.format(args.dre_net, args.dre_epochs, args.dre_lambda, args.seed, label_i, args.max_label)

        ### dre training
        dre_net_i = DR_MLP(args.dre_net, p_dropout=0.5, init_in_dim = args.num_channels*args.img_size*args.img_size)
        dre_net_i = nn.DataParallel(dre_net_i)
        #if DR model exists, then load the pretrained model; otherwise, start training the model.
        if not os.path.isfile(drefile_fullpath):
            print("\n Begin Training unconditional DR in Feature Space: >>>")
            dre_net_i, avg_train_loss = train_dre(trainloader=trainloader_dre_i, dre_net=dre_net_i, dre_precnn_net=dre_presae_encoder_net, netG=netG, net_y2h=net_y2h, path_to_ckpt=path_to_ckpt_in_train, do_label_norm=do_label_norm, do_label_denorm=do_label_denorm)
            # save model
            torch.save({
            'net_state_dict': dre_net_i.state_dict(),
            }, drefile_fullpath)
            PlotLoss(avg_train_loss, dre_loss_file_fullpath)
        else:
            # if already trained, load pre-trained DR model
            checkpoint_dre_net_i = torch.load(drefile_fullpath)
            dre_net_i.load_state_dict(checkpoint_dre_net_i['net_state_dict'])
        ##end if not
        dre_net_i = dre_net_i.cpu()
        dre_net_list.append(dre_net_i)
    ###end for i

    # Compute density ratio: function for computing a bunch of images in a numpy array
    def comp_cond_density_ratio(images, given_label, batch_size=args.samp_batch_size, dre_precnn_net=dre_presae_encoder_net, dre_net_list=dre_net_list):
        #images: a torch tensor
        #given_label: condition; unnormalized
        n_imgs = len(images)
        if batch_size>n_imgs:
            batch_size = n_imgs

        ##make sure the last iteration has enough samples
        images = torch.cat((images, images[0:batch_size]), dim=0)

        density_ratios = []
        dre_net_i = dre_net_list[unnorm_label_to_indx[given_label]] #take the density ratio model for label i
        dre_net_i = dre_net_i.cuda()
        dre_net_i.eval()
        dre_precnn_net = dre_precnn_net.cuda()
        dre_precnn_net.eval()
        # print("\n Begin computing density ratio for images >>")
        with torch.no_grad():
            n_imgs_got = 0
            while n_imgs_got < n_imgs:
                batch_images = images[n_imgs_got:(n_imgs_got+batch_size)]
                batch_images = batch_images.type(torch.float).cuda()
                batch_features = dre_precnn_net(batch_images)
                batch_ratios = dre_net_i(batch_features)
                density_ratios.append(batch_ratios.cpu().detach())
                n_imgs_got += batch_size
            ### while n_imgs_got
        density_ratios = torch.cat(density_ratios)
        density_ratios = density_ratios[0:n_imgs].numpy()

        ## back to cpu
        dre_precnn_net = dre_precnn_net.cpu()
        dre_net_i = dre_net_i.cpu()
        return density_ratios


    # Enhanced sampler based on the trained DR model
    # Rejection Sampling:"Discriminator Rejection Sampling"; based on https://github.com/shinseung428/DRS_Tensorflow/blob/master/config.py
    def fn_enhancedSampler_given_label(nfake, given_label, batch_size=args.samp_batch_size, verbose=True):
        ## given_label: unnormalized label
        ## Burn-in Stage
        n_burnin = args.samp_burnin_size
        burnin_imgs, _ = fn_sampleGAN_given_label(n_burnin, given_label, batch_size = batch_size, to_numpy=False, denorm=True)
        burnin_densityratios = comp_cond_density_ratio(burnin_imgs, given_label)
        print((burnin_densityratios.min(),np.median(burnin_densityratios),burnin_densityratios.max()))
        M_bar = np.max(burnin_densityratios)
        del burnin_imgs, burnin_densityratios; gc.collect()

        if M_bar<=1e-8: #if M_bar==0, the sampling may get stuck
            return fn_sampleGAN_given_label(nfake, given_label, batch_size = batch_size, to_numpy=True, denorm=True)
        else:
            ## Rejection sampling
            enhanced_imgs = []
            if verbose:
                pb = SimpleProgressBar()
                # pbar = tqdm(total=nfake)
            num_imgs = 0
            while num_imgs < nfake:
                batch_imgs, _ = fn_sampleGAN_given_label(batch_size, given_label, batch_size = batch_size, to_numpy=False, denorm=True)
                batch_ratios = comp_cond_density_ratio(batch_imgs, given_label)
                batch_imgs = batch_imgs.numpy() #convert to numpy array
                M_bar = np.max([M_bar, np.max(batch_ratios)])
                #threshold
                batch_p = batch_ratios/M_bar
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
            return enhanced_imgs, (given_label*np.ones(nfake)).astype(int)





###############################################################################
'''                               Evaluation                                '''
###############################################################################
if args.eval or args.samp_dump_fake_data:

    #####################
    # generate nfake images
    print("\n Start sampling {} fake images per label from GAN >>>".format(args.samp_nfake_per_label))

    ### generate fake images
    eval_labels = np.arange(1, args.max_label+1) # unnormalized labels for evaluation
    num_eval_labels = len(eval_labels)

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
                fake_images_i, fake_labels_i = fn_enhancedSampler_given_label(nfake=args.samp_nfake_per_label, given_label=eval_labels[i], batch_size=args.samp_batch_size, verbose=True)
            else:
                fake_images_i, fake_labels_i = fn_sampleGAN_given_label(nfake=args.samp_nfake_per_label, given_label=eval_labels[i], batch_size=args.samp_batch_size, to_numpy=True, denorm=True, verbose=True)
            assert fake_images_i.max()>1 and fake_images_i.max()<=255.0
            fake_images.append(fake_images_i)
            fake_labels.append(fake_labels_i.reshape(-1))
            print("\r Generated {} fake images for label {}/{}. Time elapses: {}".format(args.samp_nfake_per_label, i+1, num_eval_labels, timeit.default_timer()-start))
        ##end for i
        fake_images = np.concatenate(fake_images, axis=0)
        fake_labels = np.concatenate(fake_labels, axis=0)
        del fake_images_i, fake_labels_i; gc.collect()
        print('\n End generating fake data!')

        if args.samp_dump_fake_data:
            with h5py.File(dump_fake_images_filename, "w") as f:
                f.create_dataset('fake_images', data = fake_images, dtype='uint8', compression="gzip", compression_opts=6)
                f.create_dataset('fake_labels', data = fake_labels, dtype='int')
    else:
        print('\n Start loading generated fake data...')
        with h5py.File(dump_fake_images_filename, "r") as f:
            fake_images = f['fake_images'][:]
            fake_labels = f['fake_labels'][:]
    assert len(fake_images) == len(fake_labels)
    print("fake labels' range: MIN={}, MAX={}".format(fake_labels.min(), fake_labels.max()))
    print("End sampling!")
    print("\n We got {} fake images.".format(len(fake_images)))


    ## dump fake images for evaluation: NIQE
    if args.dump_fake_for_NIQE:
        print("\n Dumping fake images for NIQE...")
        path_to_imgs_for_NIQE = os.path.join(dump_fake_images_folder, 'fake_images_for_NIQE_CcGAN_subsampling_{}_NfakePerLabel_{}_seed_{}'.format(subsampling_method, args.samp_nfake_per_label, args.seed))
        os.makedirs(path_to_imgs_for_NIQE, exist_ok=True)
        for i in tqdm(range(len(fake_images))):
            label_i = fake_labels[i]
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
        PreNetDiversity = ResNet34_class_eval(num_classes=5, ngpu = torch.cuda.device_count()).cuda() #5 races
        checkpoint_PreNet = torch.load(args.eval_ckpt_path_Div)
        PreNetDiversity.load_state_dict(checkpoint_PreNet['net_state_dict'])

        # for LS
        PreNetLS = ResNet34_regre_eval(ngpu = torch.cuda.device_count()).cuda()
        checkpoint_PreNet = torch.load(args.eval_ckpt_path_LS)
        PreNetLS.load_state_dict(checkpoint_PreNet['net_state_dict'])


        #####################
        # normalize labels
        real_labels = do_label_norm(raw_labels)
        nfake_all = len(fake_images)
        nreal_all = len(raw_images)
        real_images = raw_images
        fake_labels = do_label_norm(fake_labels)
        

        #####################
        # Evaluate FID within a sliding window with a radius R on the label's range (i.e., [1,max_label]). The center of the sliding window locate on [R+1,2,3,...,max_label-R].
        center_start = 1+args.FID_radius
        center_stop = args.max_label-args.FID_radius
        centers_loc = np.arange(center_start, center_stop+1)
        FID_over_centers = np.zeros(len(centers_loc))
        entropies_over_centers = np.zeros(len(centers_loc)) # entropy at each center
        labelscores_over_centers = np.zeros(len(centers_loc)) #label score at each center
        num_realimgs_over_centers = np.zeros(len(centers_loc))
        for i in range(len(centers_loc)):
            center = centers_loc[i]
            interval_start = (center - args.FID_radius)/args.max_label
            interval_stop = (center + args.FID_radius)/args.max_label

            ## real images for i-th label
            indx_real = np.where((real_labels>=interval_start)*(real_labels<=interval_stop)==True)[0]
            np.random.shuffle(indx_real)
            real_images_i = real_images[indx_real]
            real_images_i = (real_images_i/255.0-0.5)/0.5
            num_realimgs_over_centers[i] = len(real_images_i)
            assert len(real_images_i)>1

            ## fake images for i-th label
            indx_fake = np.where((fake_labels>=interval_start)*(fake_labels<=interval_stop)==True)[0]
            np.random.shuffle(indx_fake)
            fake_images_i = fake_images[indx_fake]
            fake_images_i = (fake_images_i/255.0-0.5)/0.5
            fake_labels_i = fake_labels[indx_fake]
            assert len(fake_images_i)>1

            # FID
            FID_over_centers[i] = cal_FID(PreNetFID, real_images_i, fake_images_i, batch_size=args.eval_batch_size, resize = None)
            # Entropy of predicted class labels
            predicted_class_labels = predict_class_labels(PreNetDiversity, fake_images_i, batch_size=args.eval_batch_size, num_workers=args.num_workers)
            entropies_over_centers[i] = compute_entropy(predicted_class_labels)
            # Label score
            labelscores_over_centers[i], _ = cal_labelscore(PreNetLS, fake_images_i, fake_labels_i, min_label_before_shift=0, max_label_after_shift=args.max_label, batch_size = args.eval_batch_size, resize = None, num_workers=args.num_workers)

            print("\r Center:{}; Real:{}; Fake:{}; FID:{}; LS:{}; ET:{}.".format(center, len(real_images_i), len(fake_images_i), FID_over_centers[i], labelscores_over_centers[i], entropies_over_centers[i]))

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
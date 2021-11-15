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
import h5py

#----------------------------------------
from opts import gen_synth_data_opts
from utils import *
from models import *
from eval_metrics import compute_FID, compute_IS
from utils import make_z, make_y, make_fixed_z, make_fixed_y
from utils.drs import drs

#######################################################################################
'''                                   Settings                                      '''
#######################################################################################
args = gen_synth_data_opts()
print(args)


subsampling_method = "GOLD"

path_torch_home = os.path.join(args.root_path, 'torch_cache')
os.makedirs(path_torch_home, exist_ok=True)
os.environ['TORCH_HOME'] = path_torch_home

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

output_directory = os.path.join(args.root_path, 'output/Setting_{}'.format(args.gan_net))
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
## generate subset
cifar_trainset = torchvision.datasets.CIFAR10(root = os.path.join(args.data_path, 'data'), train=True, download=True)
images_train = cifar_trainset.data
images_train = np.transpose(images_train, (0, 3, 1, 2))
labels_train = np.array(cifar_trainset.targets)

cifar_testset = torchvision.datasets.CIFAR10(root = os.path.join(args.data_path, 'data'), train=False, download=True)
images_test = cifar_testset.data
images_test = np.transpose(images_test, (0, 3, 1, 2))
labels_test = np.array(cifar_testset.targets)


#######################################################################################
'''                  Load pre-trained GAN to Memory (not GPU)                       '''
#######################################################################################
if args.gan_net=="ACGAN":
    ckpt = torch.load(args.gan_ckpt_path)
    netG = ACGAN_Generator(nz=args.gan_dim_g, ny=args.num_classes)
    netG.load_state_dict(ckpt['netG_state_dict'])
    # netG = nn.DataParallel(netG)
    netD = ACGAN_Discriminator(ny=args.num_classes)
    netD.load_state_dict(ckpt['netD_state_dict'])
    # netD = nn.DataParallel(netD)
else:
    raise Exception("Not supported GAN!!")


###############################################################################
'''                             Compute FID and IS                          '''
###############################################################################
if args.eval or args.samp_dump_fake_data:
    if args.inception_from_scratch:
        #load pre-trained InceptionV3 (pretrained on CIFAR-10)
        PreNetFID = Inception3(num_classes=args.num_classes, aux_logits=True, transform_input=False)
        checkpoint_PreNet = torch.load(args.eval_ckpt_path)
        PreNetFID = nn.DataParallel(PreNetFID).cuda()
        PreNetFID.load_state_dict(checkpoint_PreNet['net_state_dict'])
    else:
        PreNetFID = inception_v3(pretrained=True, transform_input=True)
        PreNetFID = nn.DataParallel(PreNetFID).cuda()
    

    ##############################################
    ''' Compute FID between real and fake images '''
    IS_scores_all = []
    FID_scores_all = []
    Intra_FID_scores_all = []

    start = timeit.default_timer()
    for nround in range(args.samp_round):
        print("\n {}+{}, Eval round: {}/{}".format(args.gan_net, subsampling_method, nround+1, args.samp_round))

        ### generate fake images
        dump_fake_images_filename = os.path.join(dump_fake_images_folder, 'fake_images_{}_subsampling_{}_NfakePerClass_{}_seed_{}_Round_{}_of_{}.h5'.format(args.gan_net, subsampling_method, args.samp_nfake_per_class, args.seed, nround+1, args.samp_round))

        if not os.path.isfile(dump_fake_images_filename):
            print('\n Start generating fake data...')
            fake_images, fake_labels = drs(netG, netD, num_samples=args.samp_nfake_per_class, perc=args.drs_perc, nz=args.gan_dim_g, ny=args.num_classes, batch_size=args.samp_batch_size, eps=1e-6)
            print('\n End generating fake data!')

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

        ## normalize images
        assert fake_images.max()>1
        fake_images = (fake_images/255.0-0.5)/0.5
        assert images_train.max()>1
        images_train = (images_train/255.0-0.5)/0.5
        assert -1.0<=images_train.max()<=1.0 and -1.0<=images_train.min()<=1.0

        if args.eval:
            #####################
            ## Compute Intra-FID: real vs fake
            print("\n Start compute Intra-FID between real and fake images...")
            start_time = timeit.default_timer()
            intra_fid_scores = np.zeros(args.num_classes)
            for i in range(args.num_classes):
                indx_train_i = np.where(labels_train==i)[0]
                images_train_i = images_train[indx_train_i]
                indx_fake_i = np.where(fake_labels==i)[0]
                fake_images_i = fake_images[indx_fake_i]
                ##compute FID within each class
                intra_fid_scores[i] = compute_FID(PreNetFID, images_train_i, fake_images_i, batch_size = args.eval_FID_batch_size, resize = (299, 299))
                print("\r Eval round: {}/{}; Class:{}; Real:{}; Fake:{}; FID:{}; Time elapses:{}s.".format(nround+1, args.samp_round, i+1, len(images_train_i), len(fake_images_i), intra_fid_scores[i], timeit.default_timer()-start_time))
            ##end for i
            # average over all classes
            print("\n Eval round: {}/{}; Intra-FID: {}({}); min/max: {}/{}.".format(nround+1, args.samp_round, np.mean(intra_fid_scores), np.std(intra_fid_scores), np.min(intra_fid_scores), np.max(intra_fid_scores)))

            # dump FID versus class to npy
            dump_fids_filename = save_evalresults_folder + "/{}_subsampling_{}_round_{}_of_{}_fids_scratchInceptionNet_{}".format(args.gan_net, subsampling_method, nround+1, args.samp_round, args.inception_from_scratch)
            np.savez(dump_fids_filename, fids=intra_fid_scores)

            #####################
            ## Compute FID: real vs fake
            print("\n Start compute FID between real and fake images...")
            indx_shuffle_real = np.arange(len(images_train)); np.random.shuffle(indx_shuffle_real)
            indx_shuffle_fake = np.arange(len(fake_images)); np.random.shuffle(indx_shuffle_fake)
            fid_score = compute_FID(PreNetFID, images_train[indx_shuffle_real], fake_images[indx_shuffle_fake], batch_size = args.eval_FID_batch_size, resize = (299, 299))
            print("\n Eval round: {}/{}; FID between {} real and {} fake images: {}.".format(nround+1, args.samp_round, len(images_train), len(fake_images), fid_score))
            
            #####################
            ## Compute IS
            print("\n Start compute IS of fake images...")
            indx_shuffle_fake = np.arange(len(fake_images)); np.random.shuffle(indx_shuffle_fake)
            is_score, is_score_std = compute_IS(PreNetFID, fake_images[indx_shuffle_fake], batch_size = args.eval_FID_batch_size, splits=10, resize=(299,299))
            print("\n Eval round: {}/{}; IS of {} fake images: {}({}).".format(nround+1, args.samp_round, len(fake_images), is_score, is_score_std))

            #####################
            # Dump evaluation results
            eval_results_fullpath = os.path.join(save_evalresults_folder, '{}_subsampling_{}_scratchInceptionNet_{}.txt'.format(args.gan_net, subsampling_method, args.inception_from_scratch))
            if not os.path.isfile(eval_results_fullpath):
                eval_results_logging_file = open(eval_results_fullpath, "w")
                eval_results_logging_file.close()
            with open(eval_results_fullpath, 'a') as eval_results_logging_file:
                eval_results_logging_file.write("\n===================================================================================================")
                eval_results_logging_file.write("\n Separate results for {} of {} rounds; Subsampling {} \n".format(nround, args.samp_round, subsampling_method))
                print(args, file=eval_results_logging_file)
                eval_results_logging_file.write("\n Intra-FID: {}({}); min/max: {}/{}.".format(np.mean(intra_fid_scores), np.std(intra_fid_scores), np.min(intra_fid_scores), np.max(intra_fid_scores)))
                eval_results_logging_file.write("\n FID: {}.".format(fid_score))
                eval_results_logging_file.write("\n IS: {}({}).".format(is_score, is_score_std))

            ## store
            FID_scores_all.append(fid_score)
            Intra_FID_scores_all.append(np.mean(intra_fid_scores))
            IS_scores_all.append(is_score)
        ##end if args.eval
    ##end nround
    stop = timeit.default_timer()
    print("Sampling and evaluation finished! Time elapses: {}s".format(stop - start))
        
    if args.eval:

        FID_scores_all = np.array(FID_scores_all)
        Intra_FID_scores_all = np.array(Intra_FID_scores_all)
        IS_scores_all = np.array(IS_scores_all)

        #####################
        # Average Eval results
        print("\n Avg Intra-FID over {} rounds: {}({}); min/max: {}/{}.".format(args.samp_round, np.mean(Intra_FID_scores_all), np.std(Intra_FID_scores_all), np.min(Intra_FID_scores_all), np.max(Intra_FID_scores_all)))

        print("\n Avg FID over {} rounds: {}({}); min/max: {}/{}.".format(args.samp_round, np.mean(FID_scores_all), np.std(FID_scores_all), np.min(FID_scores_all), np.max(FID_scores_all)))

        print("\n Avg IS over {} rounds: {}({}); min/max: {}/{}.".format(args.samp_round, np.mean(IS_scores_all), np.std(IS_scores_all), np.min(IS_scores_all), np.max(IS_scores_all)))
        
        #####################
        # Dump evaluation results
        eval_results_fullpath = os.path.join(save_evalresults_folder, '{}_subsampling_{}_scratchInceptionNet_{}.txt'.format(args.gan_net, subsampling_method, args.inception_from_scratch))
        if not os.path.isfile(eval_results_fullpath):
            eval_results_logging_file = open(eval_results_fullpath, "w")
            eval_results_logging_file.close()
        with open(eval_results_fullpath, 'a') as eval_results_logging_file:
            eval_results_logging_file.write("\n===================================================================================================")
            eval_results_logging_file.write("\n Average results over {} rounds; Subsampling {} \n".format(args.samp_round, subsampling_method))
            print(args, file=eval_results_logging_file)
            eval_results_logging_file.write("\n Avg. Intra-FID over {} rounds: {}({}); min/max: {}/{}.".format(args.samp_round, np.mean(Intra_FID_scores_all), np.std(Intra_FID_scores_all), np.min(Intra_FID_scores_all), np.max(Intra_FID_scores_all)))
            eval_results_logging_file.write("\n Avg. FID over {} rounds: {}({}); min/max: {}/{}.".format(args.samp_round, np.mean(FID_scores_all), np.std(FID_scores_all), np.min(FID_scores_all), np.max(FID_scores_all)))
            eval_results_logging_file.write("\n Avg. IS over {} rounds: {}({}); min/max: {}/{}.".format(args.samp_round, np.mean(IS_scores_all), np.std(IS_scores_all), np.min(IS_scores_all), np.max(IS_scores_all)))
    ## if args.eval
    


# #######################################################################################
# '''               Visualize fake images of the trained GAN                          '''
# #######################################################################################
# if args.visualize_fake_images:
    
#     # First, visualize conditional generation # vertical grid
#     ## 10 rows; 10 columns (10 samples for each class)
#     n_row = args.num_classes
#     n_col = 10

#     fake_images_view = []
#     fake_labels_view = []
#     for i in range(args.num_classes):
#         fake_labels_i = i*np.ones(n_col)
#         if args.subsampling:
#             fake_images_i, _ = fn_enhancedSampler_given_label(nfake=n_col, given_label=i, batch_size=100, verbose=False)
#         else:
#             fake_images_i, _ = fn_sampleGAN_given_label(nfake=n_col, given_label=i, batch_size=100, pretrained_netG=netG, to_numpy=True)
#         fake_images_view.append(fake_images_i)
#         fake_labels_view.append(fake_labels_i)
#     ##end for i
#     fake_images_view = np.concatenate(fake_images_view, axis=0)
#     fake_labels_view = np.concatenate(fake_labels_view, axis=0)

#     ### output fake images from a trained GAN
#     filename_fake_images = save_evalresults_folder + '/{}_subsampling_{}_fake_image_grid_{}x{}.png'.format(args.gan_net, subsampling_method, n_row, n_col)
    
#     images_show = np.zeros((n_row*n_col, args.num_channels, args.img_size, args.img_size))
#     for i_row in range(n_row):
#         indx_i = np.where(fake_labels_view==i_row)[0]
#         for j_col in range(n_col):
#             curr_image = fake_images_view[indx_i[j_col]]
#             images_show[i_row*n_col+j_col,:,:,:] = curr_image
#     images_show = torch.from_numpy(images_show)
#     save_image(images_show.data, filename_fake_images, nrow=n_col, normalize=True)

# ### end if args.visualize_fake_images





print("\n ===================================================================================================")
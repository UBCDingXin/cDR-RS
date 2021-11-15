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
import random
import matplotlib.pyplot as plt
import matplotlib as mpl
from torchvision.utils import save_image
import csv
from tqdm import tqdm
import h5py
import gc

## import my stuffs
from opts import parse_opts
from utils import *
from models import *
from train_cgan import train_cgan
from eval_metrics import compute_FID, compute_IS


#######################################################################################
'''                                   Settings                                      '''
#######################################################################################
args = parse_opts()
print(args)

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
output_directory = os.path.join(args.root_path, 'output')
os.makedirs(output_directory, exist_ok=True)

save_models_folder = os.path.join(output_directory, 'saved_models')
os.makedirs(save_models_folder, exist_ok=True)

save_images_folder = os.path.join(output_directory, 'saved_images')
os.makedirs(save_images_folder, exist_ok=True)

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

print("\n Training set shape: {}x{}x{}x{}; Testing set shape: {}x{}x{}x{}.".format(images_train.shape[0], images_train.shape[1], images_train.shape[2], images_train.shape[3], images_test.shape[0], images_test.shape[1], images_test.shape[2], images_test.shape[3]))


if args.transform:
    # transform = transforms.Compose([
    #             transforms.RandomHorizontalFlip(),
    #             transforms.ToTensor(),
    #             transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5]), ##do not use other normalization constants!!!
    #             ])

    transform = transforms.Compose([
                transforms.Resize(int(args.img_size*1.1)),
                transforms.RandomCrop(args.img_size),
                transforms.RandomHorizontalFlip(),
                transforms.Resize(args.img_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5]), ##do not use other normalization constants!!!
                ])
else:
    transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5]), ##do not use other normalization constants!!!
                ])

trainset = IMGs_dataset(images_train, labels_train, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)



#######################################################################################
'''                                  Train cGAN                                     '''
#######################################################################################
filename_gan = os.path.join(save_models_folder, 'ckpt_{}_niters_{}_nDs_{}_seed_{}.pth'.format(args.gan_arch, args.niters, args.num_D_steps, args.seed))
print("\n Filename for GAN ckpt is {}".format(filename_gan))

print("\n Begin Training GAN:")
start = timeit.default_timer()
if args.gan_arch=="ACGAN":
    if not os.path.isfile(filename_gan):
        print("\n Begin training {}: ".format(args.gan_arch))
        ## model initialization
        netG = ACGAN_Generator(nz=args.dim_gan, ny=args.num_classes)
        netD = ACGAN_Discriminator(ny=args.num_classes)
        
        ## train cGAN
        netG, netD = train_cgan(trainloader, netG, netD, save_images_folder=save_images_folder, save_models_folder = save_models_folder)

        # store model
        torch.save({
            'netG_state_dict': netG.state_dict(),
            'netD_state_dict': netD.state_dict()
        }, filename_gan)
    else:
        print("\n Ckpt already exists")
        print("\n Loading...")
        checkpoint = torch.load(filename_gan)
        netG = ACGAN_Generator(nz=args.dim_gan, ny=args.num_classes)
        netG.load_state_dict(checkpoint['netG_state_dict'])
else:
    raise Exception("Not supported GAN arch...")


def fn_sampleGAN_given_label(nfake, given_label, batch_size, pretrained_netG=netG, to_numpy=True):
    raw_fake_images = []
    raw_fake_labels = []
    pretrained_netG = pretrained_netG.cuda()
    pretrained_netG.eval()
    with torch.no_grad():
        tmp = 0
        while tmp < nfake:
            z = torch.randn(batch_size, args.dim_gan, dtype=torch.float).cuda()
            labels = (given_label*torch.ones(batch_size)).type(torch.long).cuda()
            batch_fake_images = pretrained_netG(z, labels)
            raw_fake_images.append(batch_fake_images.cpu())
            raw_fake_labels.append(labels.cpu().view(-1))
            tmp += batch_size

    raw_fake_images = torch.cat(raw_fake_images, dim=0)
    raw_fake_labels = torch.cat(raw_fake_labels)

    if to_numpy:
        raw_fake_images = raw_fake_images.numpy()
        raw_fake_labels = raw_fake_labels.numpy()

    return raw_fake_images[0:nfake], raw_fake_labels[0:nfake]



###############################################################################
'''                             Compute FID and IS                          '''
###############################################################################
if args.eval_real or args.eval_fake:
    if args.inception_from_scratch:
        #load pre-trained InceptionV3 (pretrained on CIFAR-10)
        PreNetFID = Inception3(num_classes=args.num_classes, aux_logits=True, transform_input=False)
        checkpoint_PreNet = torch.load(args.eval_ckpt_path)
        PreNetFID = nn.DataParallel(PreNetFID).cuda()
        PreNetFID.load_state_dict(checkpoint_PreNet['net_state_dict'])
    else:
        PreNetFID = inception_v3(pretrained=True, transform_input=True)
        PreNetFID = nn.DataParallel(PreNetFID).cuda()
    
    ## normalize images
    images_train = (images_train/255.0-0.5)/0.5
    images_test = (images_test/255.0-0.5)/0.5

    ##############################################
    ''' Compute FID between real images as a reference '''
    if args.eval_real:
        #####################
        ## Compute Intra-FID: train vs test
        print("\n Start compute Intra-FID between real images as reference...")
        intra_fid_scores_ref = np.zeros(args.num_classes)
        for i in range(args.num_classes):
            indx_train_i = np.where(labels_train==i)[0]
            images_train_i = images_train[indx_train_i]
            indx_test_i = np.where(labels_test==i)[0]
            images_test_i = images_test[indx_test_i]
            ##compute FID within each class
            intra_fid_scores_ref[i] = compute_FID(PreNetFID, images_train_i, images_test_i, batch_size = args.FID_batch_size, resize = (299, 299))
            print("\r Class:{}; Train:{}; Test:{}; FID:{}; Time elapses:{}s.".format(i+1, len(images_train_i), len(images_test_i), intra_fid_scores_ref[i], timeit.default_timer()-start))
        ##end for i
        # average over all classes
        print("\n Ref Intra-FID: {}({}); min/max: {}/{}.".format(np.mean(intra_fid_scores_ref), np.std(intra_fid_scores_ref), np.min(intra_fid_scores_ref), np.max(intra_fid_scores_ref)))

        #####################
        ## Compute FID: train vs test
        print("\n Start compute FID between real images as reference...")
        indx_shuffle_train = np.arange(len(images_train)); np.random.shuffle(indx_shuffle_train)
        indx_shuffle_test = np.arange(len(images_test)); np.random.shuffle(indx_shuffle_test)
        fid_score_ref = compute_FID(PreNetFID, images_train[indx_shuffle_train], images_test[indx_shuffle_test], batch_size = args.FID_batch_size, resize = (299, 299))
        print("\n FID between {} training and {} test images: {}.".format(len(images_train), len(images_test), fid_score_ref))

        #####################
        ## Compute IS: train
        print("\n Start compute IS of real images as reference...")
        indx_shuffle_train = np.arange(len(images_train)); np.random.shuffle(indx_shuffle_train)
        is_score_ref, is_score_std_ref = compute_IS(PreNetFID, images_train[indx_shuffle_train], batch_size = args.FID_batch_size, splits=10, resize=(299,299))
        print("\n IS of {} training images: {}({}).".format(len(images_train), is_score_ref, is_score_std_ref))


    ##############################################
    ''' Compute FID between real and fake images '''
    if args.eval_fake or args.samp_dump_fake_data:

        IS_scores_all = []
        FID_scores_all = []
        Intra_FID_scores_all = []

        start = timeit.default_timer()
        for nround in range(args.samp_round):
            print("\n {}+{}, Eval round: {}/{}".format(args.gan_arch, "None", nround+1, args.samp_round))

            ### generate fake images
            dump_fake_images_filename = os.path.join(dump_fake_images_folder, 'fake_images_{}_niters_{}_subsampling_{}_NfakePerClass_{}_seed_{}_Round_{}_of_{}.h5'.format(args.gan_arch, args.niters, "None", args.samp_nfake_per_class, args.seed, nround+1, args.samp_round))

            if not os.path.isfile(dump_fake_images_filename):
                print('\n Start generating fake data...')
                fake_images = []
                fake_labels = []
                for i in range(args.num_classes):
                    print("\n Generate {} fake images for class {}/{}.".format(args.samp_nfake_per_class, i+1, args.num_classes))
                    fake_images_i, fake_labels_i = fn_sampleGAN_given_label(nfake=args.samp_nfake_per_class, given_label=i, batch_size=args.samp_batch_size, pretrained_netG=netG, to_numpy=True)
                    assert fake_images_i.max()<=1 and fake_images_i.min()>=-1
                    ## denormalize images to save memory
                    fake_images_i = (fake_images_i*0.5+0.5)*255.0
                    fake_images_i = fake_images_i.astype(np.uint8)
                    assert fake_images_i.max()>1 and fake_images_i.max()<=255.0
                    fake_images.append(fake_images_i)
                    fake_labels.append(fake_labels_i.reshape(-1))
                fake_images = np.concatenate(fake_images, axis=0)
                fake_labels = np.concatenate(fake_labels, axis=0)
                del fake_images_i, fake_labels_i; gc.collect()
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

            assert fake_images.max()>1
            fake_images = (fake_images/255.0-0.5)/0.5
            assert -1.0<=images_train.max()<=1.0 and -1.0<=images_train.min()<=1.0

            if args.eval_fake:
                #####################
                ## Compute Intra-FID: real vs fake
                print("\n Start compute Intra-FID between real and fake images...")
                intra_fid_scores = np.zeros(args.num_classes)
                for i in range(args.num_classes):
                    indx_train_i = np.where(labels_train==i)[0]
                    images_train_i = images_train[indx_train_i]
                    indx_fake_i = np.where(fake_labels==i)[0]
                    fake_images_i = fake_images[indx_fake_i]
                    ##compute FID within each class
                    intra_fid_scores[i] = compute_FID(PreNetFID, images_train_i, fake_images_i, batch_size = args.FID_batch_size, resize = (299, 299))
                    print("\r Eval round: {}/{}; Class:{}; Real:{}; Fake:{}; FID:{}; Time elapses:{}s.".format(nround+1, args.samp_round, i+1, len(images_train_i), len(fake_images_i), intra_fid_scores[i], timeit.default_timer()-start))
                ##end for i
                # average over all classes
                print("\n Eval round: {}/{}; Intra-FID: {}({}); min/max: {}/{}.".format(nround+1, args.samp_round, np.mean(intra_fid_scores), np.std(intra_fid_scores), np.min(intra_fid_scores), np.max(intra_fid_scores)))

                # dump FID versus class to npy
                dump_fids_filename = save_evalresults_folder + "/{}_niters_{}_subsampling_{}_round_{}_of_{}_fids_scratchInceptionNet_{}".format(args.gan_arch, args.niters, "None", nround+1, args.samp_round, args.inception_from_scratch)
                np.savez(dump_fids_filename, fids=intra_fid_scores)

                #####################
                ## Compute FID: real vs fake
                print("\n Start compute FID between real and fake images...")
                indx_shuffle_real = np.arange(len(images_train)); np.random.shuffle(indx_shuffle_real)
                indx_shuffle_fake = np.arange(len(fake_images)); np.random.shuffle(indx_shuffle_fake)
                fid_score = compute_FID(PreNetFID, images_train[indx_shuffle_real], fake_images[indx_shuffle_fake], batch_size = args.FID_batch_size, resize = (299, 299))
                print("\n Eval round: {}/{}; FID between {} real and {} fake images: {}.".format(nround+1, args.samp_round, len(images_train), len(fake_images), fid_score))
                
                #####################
                ## Compute IS
                print("\n Start compute IS of fake images...")
                indx_shuffle_fake = np.arange(len(fake_images)); np.random.shuffle(indx_shuffle_fake)
                is_score, is_score_std = compute_IS(PreNetFID, fake_images[indx_shuffle_fake], batch_size = args.FID_batch_size, splits=10, resize=(299,299))
                print("\n Eval round: {}/{}; IS of {} fake images: {}({}).".format(nround+1, args.samp_round, len(fake_images), is_score, is_score_std))

                ## store
                FID_scores_all.append(fid_score)
                Intra_FID_scores_all.append(np.mean(intra_fid_scores))
                IS_scores_all.append(is_score)
        ##end nround
        stop = timeit.default_timer()
        print("Sampling and evaluation finished! Time elapses: {}s".format(stop - start))
        
        if args.eval_fake:

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
            eval_results_fullpath = os.path.join(save_evalresults_folder, '{}_niters_{}_subsampling_{}_scratchInceptionNet_{}.txt'.format(args.gan_arch, args.niters, "None", args.inception_from_scratch))
            if not os.path.isfile(eval_results_fullpath):
                eval_results_logging_file = open(eval_results_fullpath, "w")
                eval_results_logging_file.close()
            with open(eval_results_fullpath, 'a') as eval_results_logging_file:
                eval_results_logging_file.write("\n===================================================================================================")
                eval_results_logging_file.write("\n Subsampling {} \n".format("None"))
                print(args, file=eval_results_logging_file)
                eval_results_logging_file.write("\n Avg. Intra-FID over {} rounds: {}({}); min/max: {}/{}.".format(args.samp_round, np.mean(Intra_FID_scores_all), np.std(Intra_FID_scores_all), np.min(Intra_FID_scores_all), np.max(Intra_FID_scores_all)))
                eval_results_logging_file.write("\n Avg. FID over {} rounds: {}({}); min/max: {}/{}.".format(args.samp_round, np.mean(FID_scores_all), np.std(FID_scores_all), np.min(FID_scores_all), np.max(FID_scores_all)))
                eval_results_logging_file.write("\n Avg. IS over {} rounds: {}({}); min/max: {}/{}.".format(args.samp_round, np.mean(IS_scores_all), np.std(IS_scores_all), np.min(IS_scores_all), np.max(IS_scores_all)))



#######################################################################################
'''               Visualize fake images of the trained GAN                          '''
#######################################################################################
if args.visualize_fake_images:

    # First, visualize conditional generation # vertical grid
    ## 10 rows; 10 columns (10 samples for each class)
    n_row = args.num_classes
    n_col = 10

    fake_images_view = []
    fake_labels_view = []
    for i in range(args.num_classes):
        fake_labels_i = i*np.ones(n_col)
        # if args.subsampling:
        #     fake_images_i, _ = fn_enhancedSampler_given_label(nfake=n_col, given_label=i, batch_size=100, verbose=False)
        # else:
        #     fake_images_i, _ = fn_sampleGAN_given_label(nfake=n_col, given_label=i, batch_size=100, pretrained_netG=netG, to_numpy=True)
        fake_images_i, _ = fn_sampleGAN_given_label(nfake=n_col, given_label=i, batch_size=100, pretrained_netG=netG, to_numpy=True)
        fake_images_view.append(fake_images_i)
        fake_labels_view.append(fake_labels_i)
    ##end for i
    fake_images_view = np.concatenate(fake_images_view, axis=0)
    fake_labels_view = np.concatenate(fake_labels_view, axis=0)

    ### output fake images from a trained GAN
    filename_fake_images = save_evalresults_folder + '/{}_niters_{}_subsampling_{}_fake_image_grid_{}x{}.png'.format(args.gan_arch, args.niters, "None", n_row, n_col)
    
    images_show = np.zeros((n_row*n_col, args.num_channels, args.img_size, args.img_size))
    for i_row in range(n_row):
        indx_i = np.where(fake_labels_view==i_row)[0]
        for j_col in range(n_col):
            curr_image = fake_images_view[indx_i[j_col]]
            images_show[i_row*n_col+j_col,:,:,:] = curr_image
    images_show = torch.from_numpy(images_show)
    save_image(images_show.data, filename_fake_images, nrow=n_col, normalize=True)

### end if args.visualize_fake_images



print("\n ===================================================================================================")
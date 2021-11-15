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


#######################################################################################
'''                                   Settings                                      '''
#######################################################################################
args = gen_synth_data_opts()
print(args)


subsampling_method = "DDLS_nSteps_{}_alpha_{}_stepLr_{}_epsStd_{}".format(args.ddls_n_steps, args.ddls_alpha, args.ddls_step_lr, args.ddls_eps_std)

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
output_directory = os.path.join(args.root_path, 'output/Setting_{}'.format(args.gan_net))
os.makedirs(output_directory, exist_ok=True)

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
if args.gan_net=="BigGAN":
    ckpt_g = torch.load(args.gan_gene_ckpt_path)
    ckpt_d = torch.load(args.gan_disc_ckpt_path)
    netG = BigGAN_Generator(dim_z=args.gan_dim_g, resolution=args.img_size, G_attn='0', n_classes=args.num_classes, G_shared=False)
    netG.load_state_dict(ckpt_g)
    netG = nn.DataParallel(netG)
    netD = BigGAN_Discriminator(resolution=args.img_size, D_attn='0', n_classes=args.num_classes)
    netD.load_state_dict(ckpt_d)
    netD = nn.DataParallel(netD)
elif args.gan_net=="SNGAN":
    ckpt = torch.load(args.gan_gene_ckpt_path)
    netG = SNGAN_Generator(dim_z=args.gan_dim_g, num_classes=args.num_classes)
    netG.load_state_dict(ckpt['netG_state_dict'])
    netG = nn.DataParallel(netG)
    netD = SNGAN_Discriminator(num_classes=args.num_classes)
    netD.load_state_dict(ckpt['netD_state_dict'])
    netD = nn.DataParallel(netD)
else:
    raise Exception("Not supported GAN!!")



### Langevin dynamics
### refer to DDLS's code at https://papers.nips.cc/paper/2020/hash/90525e70b7842930586545c6f1c9310c-Abstract.html
def e_grad(z, given_label, netG, netD, alpha, ret_e=False):
    
    batch_size = z.shape[0]
    z_dim = z.shape[1]
    z = autograd.Variable(z, requires_grad=True)
    ## prior proposal for z
    prior_mean = torch.zeros(z_dim).cuda()
    prior_covm = torch.eye(z_dim).cuda()
    prior_z = torch.distributions.multivariate_normal.MultivariateNormal(prior_mean, prior_covm)
    ## compute energy
    # logp_z = torch.sum(prior_z.log_prob(z), dim=1, keepdim=True)
    logp_z = prior_z.log_prob(z).view(-1,1)
    gen_labels = (given_label*torch.ones(batch_size)).type(torch.long).cuda()
    disc = netD(netG(z, gen_labels), gen_labels)
    Energy = - logp_z - alpha * disc
    # gradients = autograd.grad(outputs=Energy, inputs=z,
    #                           grad_outputs=torch.ones_like(Energy).cuda(),
    #                           create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = autograd.grad(outputs=Energy, inputs=z,
                              grad_outputs=torch.ones_like(Energy).cuda())[0]
    if ret_e:
        return Energy, gradients
    return gradients


def langevin_dynamics(z, given_label, netG, netD, alpha=args.ddls_alpha, n_steps=args.ddls_n_steps, step_lr=args.ddls_step_lr, eps_std=args.ddls_eps_std):
    z_sp = []
    batch_size, z_dim = z.shape
    for _ in range(n_steps):
        if _ % 10 == 0:
            z_sp.append(z)
        eps = eps_std * torch.randn((batch_size,z_dim)).type(torch.float).cuda()
        gradients = e_grad(z, given_label, netG, netD, alpha, ret_e=False)
        # z = z - step_lr * gradients[0] + eps
        assert gradients.shape == z.shape
        z = z - step_lr * gradients + eps
    z_sp.append(z)
    # print(n_steps, len(z_sp), z.shape)
    return z_sp


def langevin_sample(given_label, netG, netD, nfake=10000, batch_size=100, verbose=True):
    netG = netG.cuda()
    netD = netD.cuda()
    netG.eval()
    netD.eval()

    fake_images = []
    if verbose:
        pb = SimpleProgressBar()
    num_taken = 0
    while num_taken < nfake:
        z = torch.randn(batch_size, args.gan_dim_g, dtype=torch.float).cuda()
        z_sp = langevin_dynamics(z, given_label, netG, netD)
        batch_labels = (given_label*torch.ones(batch_size)).type(torch.long).cuda()
        batch_images = netG(z_sp[-1], batch_labels)
        batch_images = batch_images.detach().cpu().numpy()
        # batch_images = ((batch_images*0.5+0.5)*255.0).astype(np.uint8) ##denorm
        fake_images.append(batch_images)
        num_taken+=len(batch_images)
        if verbose:
            pb.update(min(float(num_taken)/nfake, 1)*100)

    fake_images = np.concatenate(fake_images, axis=0)
    fake_labels = (given_label*torch.ones(len(fake_images))).type(torch.long).view(-1)
    return fake_images[0:nfake], fake_labels[0:nfake]




###############################################################################
'''                             Compute FID and IS                          '''
###############################################################################
if args.eval or args.samp_dump_fake_data:
    if args.inception_from_scratch:
        #load pre-trained InceptionV3 (pretrained on CIFAR-100)
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

        # ### generate fake images; one h5 file
        # dump_fake_images_filename = os.path.join(dump_fake_images_folder, 'fake_images_{}_subsampling_{}_NfakePerClass_{}_seed_{}_Round_{}_of_{}.h5'.format(args.gan_net, subsampling_method, args.samp_nfake_per_class, args.seed, nround+1, args.samp_round))

        # if not os.path.isfile(dump_fake_images_filename):
        #     print('\n Start generating fake data...')
        #     fake_images = []
        #     fake_labels = []
        #     for i in range(args.num_classes):
        #         print("\n Generate {} fake images for class {}/{}.".format(args.samp_nfake_per_class, i+1, args.num_classes))
        #         fake_images_i, fake_labels_i = langevin_sample(given_label=i, netG=netG, netD=netD, nfake=args.samp_nfake_per_class, batch_size=args.samp_batch_size)
        #         assert fake_images_i.max()<=1 and fake_images_i.min()>=-1
        #         ## denormalize images to save memory
        #         fake_images_i = (fake_images_i*0.5+0.5)*255.0
        #         fake_images_i = fake_images_i.astype(np.uint8)
        #         assert fake_images_i.max()>1 and fake_images_i.max()<=255.0
        #         fake_images.append(fake_images_i)
        #         fake_labels.append(fake_labels_i.reshape(-1))
        #     fake_images = np.concatenate(fake_images, axis=0)
        #     fake_labels = np.concatenate(fake_labels, axis=0)
        #     del fake_images_i, fake_labels_i; gc.collect()
        #     print('\n End generating fake data!')

        #     if args.samp_dump_fake_data:
        #         with h5py.File(dump_fake_images_filename, "w") as f:
        #             f.create_dataset('fake_images', data = fake_images, dtype='uint8', compression="gzip", compression_opts=6)
        #             f.create_dataset('fake_labels', data = fake_labels, dtype='float')
        # else:
        #     print('\n Start loading generated fake data...')
        #     with h5py.File(dump_fake_images_filename, "r") as f:
        #         fake_images = f['fake_images'][:]
        #         fake_labels = f['fake_labels'][:]
        # assert len(fake_images) == len(fake_labels)



        ### generate fake images; separate h5 files
        dump_fake_images_folder_nround = os.path.join(dump_fake_images_folder, 'fake_images_{}_subsampling_{}_NfakePerClass_{}_seed_{}_Round_{}_of_{}'.format(args.gan_net, subsampling_method, args.samp_nfake_per_class, args.seed, nround+1, args.samp_round))
        os.makedirs(dump_fake_images_folder_nround, exist_ok=True)

        fake_images = []
        fake_labels = []
        for i in range(args.num_classes):
            dump_fake_images_filename = os.path.join(dump_fake_images_folder_nround, 'class_{}_of_{}.h5'.format(i+1,args.num_classes))

            if not os.path.isfile(dump_fake_images_filename):
                print("\n Start generating {} fake images for class {}/{}.".format(args.samp_nfake_per_class, i+1, args.num_classes))
                fake_images_i, fake_labels_i = langevin_sample(given_label=i, netG=netG, netD=netD, nfake=args.samp_nfake_per_class, batch_size=args.samp_batch_size)
                ## denormalize images to save memory
                assert fake_images_i.max()<=1 and fake_images_i.min()>=-1
                fake_images_i = (fake_images_i*0.5+0.5)*255.0
                fake_images_i = fake_images_i.astype(np.uint8)

                if args.samp_dump_fake_data:
                    with h5py.File(dump_fake_images_filename, "w") as f:
                        f.create_dataset('fake_images_i', data = fake_images_i, dtype='uint8', compression="gzip", compression_opts=6)
                        f.create_dataset('fake_labels_i', data = fake_labels_i, dtype='float')

            else:
                print('\n Start loading generated fake data for class {}/{}...'.format(i+1,args.num_classes))
                with h5py.File(dump_fake_images_filename, "r") as f:
                    fake_images_i = f['fake_images_i'][:]
                    fake_labels_i = f['fake_labels_i'][:]
            
            assert fake_images_i.max()>1 and fake_images_i.max()<=255.0
            fake_images.append(fake_images_i)
            fake_labels.append(fake_labels_i.reshape(-1))
        ##end for i
        fake_images = np.concatenate(fake_images, axis=0)
        fake_labels = np.concatenate(fake_labels)



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
        fake_images_i, _ = langevin_sample(given_label=i, netG=netG, netD=netD, nfake=n_col, batch_size=100)
        fake_images_view.append(fake_images_i)
        fake_labels_view.append(fake_labels_i)
    ##end for i
    fake_images_view = np.concatenate(fake_images_view, axis=0)
    fake_labels_view = np.concatenate(fake_labels_view, axis=0)

    ### output fake images from a trained GAN
    filename_fake_images = save_evalresults_folder + '/{}_subsampling_{}_fake_image_grid_{}x{}.png'.format(args.gan_net, subsampling_method, n_row, n_col)
    
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
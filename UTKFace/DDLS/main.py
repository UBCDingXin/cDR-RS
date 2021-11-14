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
from eval_metrics import cal_FID, cal_labelscore



#######################################################################################
'''                                   Settings                                      '''
#######################################################################################
args = gen_synth_data_opts()
print(args)


subsampling_method = "DDLS_nSteps_{}_alpha_{}_stepLr_{}_epsStd_{}".format(args.ddls_n_steps, args.ddls_alpha, args.ddls_step_lr, args.ddls_eps_std)

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



### Langevin dynamics
### refer to DDLS's code at https://papers.nips.cc/paper/2020/hash/90525e70b7842930586545c6f1c9310c-Abstract.html
def e_grad(z, given_label, netG, netD, net_y2h, alpha, ret_e=False):
    
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
    gen_labels = (given_label*torch.ones(batch_size)).type(torch.float).cuda()
    disc = netD(netG(z, net_y2h(gen_labels).detach()), net_y2h(gen_labels).detach())
    Energy = - logp_z - alpha * disc
    # gradients = autograd.grad(outputs=Energy, inputs=z,
    #                           grad_outputs=torch.ones_like(Energy).cuda(),
    #                           create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = autograd.grad(outputs=Energy, inputs=z,
                              grad_outputs=torch.ones_like(Energy).cuda())[0]
    if ret_e:
        return Energy, gradients
    return gradients


def langevin_dynamics(z, given_label, netG, netD, net_y2h, alpha=args.ddls_alpha, n_steps=args.ddls_n_steps, step_lr=args.ddls_step_lr, eps_std=args.ddls_eps_std):
    z_sp = []
    batch_size, z_dim = z.shape
    for step in range(n_steps):
        if step % 10 == 0:
            z_sp.append(z)
        eps = eps_std * torch.randn((batch_size,z_dim)).type(torch.float).cuda()
        gradients = e_grad(z, given_label, netG, netD, net_y2h, alpha, ret_e=False)
        # z = z - step_lr * gradients[0] + eps
        assert gradients.shape == z.shape
        z = z - step_lr * gradients + eps
        # print(step, n_steps)
    z_sp.append(z)
    # print(n_steps, len(z_sp), z.shape)
    return z_sp


def langevin_sample(given_label, netG, netD, net_y2h, nfake=1000, batch_size=100, verbose=True):
    ''' 
    given_label: normalized
    '''

    assert 0<=given_label<=1

    netG = netG.cuda()
    netD = netD.cuda()
    net_y2h = net_y2h.cuda()
    netG.eval()
    netD.eval()
    net_y2h.eval()

    fake_images = []
    if verbose:
        pb = SimpleProgressBar()
    num_taken = 0
    while num_taken < nfake:
        z = torch.randn(batch_size, args.gan_dim_g, dtype=torch.float).cuda()
        z_sp = langevin_dynamics(z, given_label, netG, netD, net_y2h)
        batch_labels = (given_label*torch.ones(batch_size)).type(torch.float).cuda()
        batch_images = netG(z_sp[-1], net_y2h(batch_labels).detach())
        batch_images = batch_images.detach().cpu().numpy()
        batch_images = ((batch_images*0.5+0.5)*255.0).astype(np.uint8) ##denorm
        fake_images.append(batch_images)
        num_taken+=len(batch_images)
        if verbose:
            pb.update(min(float(num_taken)/nfake, 1)*100)

    fake_images = np.concatenate(fake_images, axis=0)
    fake_labels = (given_label*np.ones(len(fake_images))).astype(float).reshape(-1)
    return fake_images[0:nfake], fake_labels[0:nfake]




###############################################################################
'''                               Evaluation                                '''
###############################################################################
if args.eval or args.samp_dump_fake_data:
    
    print("\n Start sampling {} fake images per label from GAN >>>".format(args.samp_nfake_per_label))
    ### generate fake images
    eval_labels = np.arange(1, args.max_label+1) # unnormalized labels for evaluation
    num_eval_labels = len(eval_labels)
    eval_labels_norm = eval_labels/args.max_label


    # #####################
    # # generate nfake images; one h5 file
    # dump_fake_images_filename = os.path.join(dump_fake_images_folder, 'fake_images_CcGAN_subsampling_{}_nfakePerLabel_{}_nfake_{}_seed_{}.h5'.format(subsampling_method, args.samp_nfake_per_label, int(num_eval_labels*args.samp_nfake_per_label), args.seed))
    # print('\n '+dump_fake_images_filename)

    # if not os.path.isfile(dump_fake_images_filename):
    #     print('\n Start generating fake data...')
    #     fake_images = []
    #     fake_labels = []
    #     start = timeit.default_timer()
    #     for i in range(num_eval_labels):
    #         print("\n Start generating {} fake images for label {}/{}.".format(args.samp_nfake_per_label, i+1, num_eval_labels))
    #         fake_images_i, fake_labels_i = langevin_sample(given_label=eval_labels_norm[i], netG=netG, netD=netD, net_y2h=net_y2h, nfake=args.samp_nfake_per_label, batch_size=args.samp_batch_size, verbose=True)
    #         assert fake_images_i.max()>1 and fake_images_i.max()<=255.0
    #         fake_images.append(fake_images_i)
    #         fake_labels.append(fake_labels_i.reshape(-1))
    #         print("\r Generated {} fake images for label {}/{}. Time elapses: {}".format(args.samp_nfake_per_label, i+1, num_eval_labels, timeit.default_timer()-start))
    #     ##end for i
    #     fake_images = np.concatenate(fake_images, axis=0)
    #     fake_labels = np.concatenate(fake_labels, axis=0)
    #     del fake_images_i, fake_labels_i; gc.collect()
    #     print('\n End generating fake data!')
    #     fake_labels = (fake_labels*args.max_label).astype(int) #denormalize fake labels

    #     if args.samp_dump_fake_data:
    #         with h5py.File(dump_fake_images_filename, "w") as f:
    #             f.create_dataset('fake_images', data = fake_images, dtype='uint8', compression="gzip", compression_opts=6)
    #             f.create_dataset('fake_labels', data = fake_labels, dtype='int')
    # else:
    #     print('\n Start loading generated fake data...')
    #     with h5py.File(dump_fake_images_filename, "r") as f:
    #         fake_images = f['fake_images'][:]
    #         fake_labels = f['fake_labels'][:]
    # assert len(fake_images) == len(fake_labels)
    # print("fake labels' range: {} to {}".format(fake_labels.min(), fake_labels.max()))
    # print("End sampling!")
    # print("\n We got {} fake images.".format(len(fake_images)))


    #####################
    ### generate fake images; separate h5 files
    dump_fake_images_folder = os.path.join(dump_fake_images_folder, 'fake_images_CcGAN_subsampling_{}_nfakePerLabel_{}_nfake_{}_seed_{}'.format(subsampling_method, args.samp_nfake_per_label, int(num_eval_labels*args.samp_nfake_per_label), args.seed))
    os.makedirs(dump_fake_images_folder, exist_ok=True)

    fake_images = []
    fake_labels = []
    start = timeit.default_timer()
    for i in range(num_eval_labels):
        dump_fake_images_filename = os.path.join(dump_fake_images_folder, 'label_{}_of_{}.h5'.format(i+1,num_eval_labels))

        if not os.path.isfile(dump_fake_images_filename):
            print("\n Start generating {} fake images for label {}/{}.".format(args.samp_nfake_per_label, i+1, num_eval_labels))
            fake_images_i, fake_labels_i = langevin_sample(given_label=eval_labels_norm[i], netG=netG, netD=netD, net_y2h=net_y2h, nfake=args.samp_nfake_per_label, batch_size=args.samp_batch_size, verbose=True)
            assert fake_images_i.max()>1 and fake_images_i.max()<=255.0
            assert fake_labels_i.max()<=1 and fake_labels_i.min()>=0
            fake_labels_i = (fake_labels_i*args.max_label).astype(int) #denormalize fake labels
            
            print("\r Generated {} fake images for label {}/{}. Time elapses: {}".format(args.samp_nfake_per_label, i+1, num_eval_labels, timeit.default_timer()-start))
            
            if args.samp_dump_fake_data:
                with h5py.File(dump_fake_images_filename, "w") as f:
                    f.create_dataset('fake_images_i', data = fake_images_i, dtype='uint8', compression="gzip", compression_opts=6)
                    f.create_dataset('fake_labels_i', data = fake_labels_i, dtype='int')
        else:
            print('\n Start loading generated fake data for label {}/{}...'.format(i+1,num_eval_labels))
            with h5py.File(dump_fake_images_filename, "r") as f:
                fake_images_i = f['fake_images_i'][:]
                fake_labels_i = f['fake_labels_i'][:]
        
        assert fake_images_i.max()>1 and fake_images_i.max()<=255.0
        fake_images.append(fake_images_i)
        fake_labels.append(fake_labels_i.reshape(-1))
    ##end for i
    fake_images = np.concatenate(fake_images, axis=0)
    fake_labels = np.concatenate(fake_labels)
    assert len(fake_images) == len(fake_labels)
    print("fake labels' range: {} to {}".format(fake_labels.min(), fake_labels.max()))
    print("End sampling!")
    print("\n We got {} fake images.".format(len(fake_images)))



    ## dump fake images for evaluation: NIQE
    if args.dump_fake_for_NIQE:
        print("\n Dumping fake images for NIQE...")
        path_to_imgs_for_NIQE = os.path.join(dump_fake_images_folder, 'fake_images_for_NIQE_CcGAN_subsampling_{}_NfakePerLabel_{}_seed_{}'.format(subsampling_method, args.samp_nfake_per_label, args.seed))
        os.makedirs(path_to_imgs_for_NIQE, exist_ok=True)
        for i in tqdm(range(len(fake_images))):
            label_i = int(fake_labels[i])
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
        real_labels = raw_labels/args.max_label
        nfake_all = len(fake_images)
        nreal_all = len(raw_images)
        real_images = raw_images
        fake_labels = fake_labels/args.max_label
        

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
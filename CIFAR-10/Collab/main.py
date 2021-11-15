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


subsampling_method = "Collab"

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


transform_dre = transforms.Compose([
                # transforms.RandomCrop((args.img_size, args.img_size), padding=4), ## note that some GAN training does not involve cropping!!!
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5]), ##do not use other normalization constants!!!
                ])


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





#########################################
''' Discriminator Shaping '''

def delta_size(batch_size, gan_net=args.gan_net):
    if gan_net=="BigGAN":
        return (batch_size, 256, 16, 16)
    elif gan_net=="SNGAN":
        return (batch_size, 128, 16, 16)
    else:
        raise Exception("Not supported GAN!!")


if args.disc_shaping:
    print("Start discriminator shaping...")

    netG = netG.cuda()
    netD = netD.cuda()
    netG.eval()

    optim_d = optim.SGD(netD.parameters(), lr=args.lr_d)
    criterion = nn.BCEWithLogitsLoss()

    pbar = tqdm(range(args.niter))
    for i in pbar:

        # probabilistic refinement 
        proba_refine = torch.zeros(delta_size(args.batch_size, gan_net=args.gan_net), dtype=torch.float32, requires_grad=False, device="cuda") ##add to a hidden map of G
        proba_steps = torch.LongTensor(args.batch_size,1).random_() % args.rollout_steps
        proba_steps_one_hot = torch.LongTensor(args.batch_size, args.rollout_steps)
        proba_steps_one_hot.zero_()
        proba_steps_one_hot.scatter_(1, proba_steps, 1)

        delta_refine = torch.zeros(delta_size(args.batch_size, gan_net=args.gan_net), dtype=torch.float32, requires_grad=True, device="cuda")
        optim_r = optim.Adam([delta_refine], lr=args.rollout_rate)
        adv_label = torch.full((args.batch_size,), 1, dtype=torch.float).cuda()

        # synthesize refined samples
        noise_batch = torch.randn(args.batch_size, args.gan_dim_g, dtype=torch.float).cuda()
        batch_labels = np.random.choice(np.arange(args.num_classes), size=args.batch_size, replace=True) #randomly generated
        batch_labels = torch.from_numpy(batch_labels).type(torch.long).cuda()
        
        netD.eval()
        for k in range(args.rollout_steps):
            optim_r.zero_grad()
            batch_fake_images = netG(noise_batch, batch_labels, delta_refine) ##delta_refine is added to a hidden map of G
            output = netD(batch_fake_images.detach(), batch_labels)
            loss_r = criterion(output.view(-1,1), adv_label.view(-1,1))
            loss_r.backward()
            optim_r.step()
            # probabilistic assignment
            proba_refine[proba_steps_one_hot[:,k] == 1, :] = delta_refine[proba_steps_one_hot[:,k] == 1, :]
        ##end for k


        ## Shape D network: maximize log(D(x)) + log(1 - D(R(G(z))))
        netD.train()
        optim_d.zero_grad()

        # train with real
        batch_unique_labels = np.sort(np.array(list(set(batch_labels.view(-1).cpu().numpy()))))
        batch_real_images = []
        for label_i in batch_unique_labels:
            num_i = len(np.where(batch_labels.view(-1).cpu().numpy()==label_i)[0]) #number of fake images in the batch with label_i
            indx_label_i = np.where(labels_train==label_i)[0]
            indx_label_i = np.random.choice(indx_label_i, size=num_i, replace=True)
            batch_real_images.append(images_train[indx_label_i])
        batch_real_images = np.concatenate(batch_real_images, axis=0)
        trainset = IMGs_dataset(batch_real_images, labels=None, transform=transform_dre)
        train_dataloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=False)
        train_dataloader = iter(train_dataloader)
        batch_real_images = train_dataloader.next()
        batch_real_images = batch_real_images.type(torch.float).cuda()
        assert batch_real_images.max().item()<=1

        output = netD(batch_real_images, batch_labels)
        loss_d_real = criterion(output.view(-1,1), adv_label.view(-1,1))
        loss_d_real.backward()

        # train with refined
        adv_label.fill_(0)
        batch_fake_images = netG(noise_batch, batch_labels, proba_refine)
        output = netD(batch_fake_images.detach(), batch_labels)
        loss_d_fake = criterion(output.view(-1,1), adv_label.view(-1,1))
        loss_d_fake.backward()

        loss_d = loss_d_real + loss_d_fake
        optim_d.step()

        # if i % 500 == 0:
        #     # Display samples
        #     print('[%d/%d] Loss_D: %.4f' % (i, args.niter, loss_d.item()))

        pbar.set_description("[{}/{}] Loss_D:{:.4f}".format(i,args.niter, loss_d.item()))

    netG = netG.cpu()
    netD = netD.cpu()
    del optim_d, optim_r; gc.collect()



#########################################
''' Collaborative Sampling '''
def collab_sampling(given_label, netG, netD, nfake=10000, batch_size=100, verbose=True):
    netG = netG.cuda()
    netD = netD.cuda()
    netG.eval()
    netD.eval()

    criterion = nn.BCEWithLogitsLoss()

    fake_images = []
    if verbose:
        pb = SimpleProgressBar()
    num_taken = 0
    while num_taken < nfake:
        z = torch.randn(batch_size, args.gan_dim_g, dtype=torch.float).cuda()
        batch_labels = (given_label*torch.ones(batch_size)).type(torch.long).cuda()

        delta_refine = torch.zeros(delta_size(batch_size, gan_net=args.gan_net), dtype=torch.float32, requires_grad=True, device="cuda")
        optim_r = optim.Adam([delta_refine], lr=args.rollout_rate)
        avd_label = torch.full((batch_size,), 1, dtype=torch.float).cuda()
        for k in range(args.rollout_steps):
            optim_r.zero_grad()
            batch_fake_images = (netG(z, batch_labels, delta_refine)).detach()
            output = netD(batch_fake_images, batch_labels)
            loss_r = criterion(output.view(-1,1), avd_label.view(-1,1))
            loss_r.backward()
            optim_r.step()
        batch_fake_images = (netG(z, batch_labels, delta_refine)).detach().cpu().numpy()
        fake_images.append(batch_fake_images)
        num_taken+=len(batch_fake_images)
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
        #         fake_images_i, fake_labels_i = collab_sampling(given_label=i, netG=netG, netD=netD, nfake=args.samp_nfake_per_class, batch_size=args.samp_batch_size)
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
                fake_images_i, fake_labels_i = collab_sampling(given_label=i, netG=netG, netD=netD, nfake=args.samp_nfake_per_class, batch_size=args.samp_batch_size)
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
        fake_images_i, _ = collab_sampling(given_label=i, netG=netG, netD=netD, nfake=n_col, batch_size=100, verbose=False)
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
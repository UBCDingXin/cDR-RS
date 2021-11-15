import argparse

def parse_opts():
    parser = argparse.ArgumentParser()

    ''' Overall Settings '''
    parser.add_argument('--root_path', type=str, default='')
    parser.add_argument('--data_path', type=str, default='')
    parser.add_argument('--eval_ckpt_path', type=str, default='')
    parser.add_argument('--seed', type=int, default=2021)

    ''' Dataset '''
    parser.add_argument('--num_channels', type=int, default=3, metavar='N')
    parser.add_argument('--img_size', type=int, default=32, metavar='N')
    parser.add_argument('--num_classes', type=int, default=10, metavar='N',)
    parser.add_argument('--show_real_imgs', action='store_true', default=False)
    parser.add_argument('--visualize_fake_images', action='store_true', default=False)

    ''' GAN Training '''
    parser.add_argument('--gan_arch', type=str, default='ACGAN')
    parser.add_argument('--niters', type=int, default=100000, metavar='N',
                        help='number of epochs to train CNNs (default: 200)')
    parser.add_argument('--resume_niter', type=int, default=0, metavar='N',
                        help='resume training.')
    parser.add_argument('--save_freq', type=int, default=500, metavar='N',
                        help='freq to save ckpt.')
    parser.add_argument('--batch_size', type=int, default=512, metavar='N',
                        help='input batch size for training')
    parser.add_argument('--dim_gan', type=int, default=128,
                        help='Latent dimension of GAN')
    parser.add_argument('--lr_g', type=float, default=1e-4,
                        help='learning rate for generator')
    parser.add_argument('--lr_d', type=float, default=1e-4,
                        help='learning rate for generator')
    parser.add_argument('--num_D_steps', type=int, default=1, 
                        help='number of Ds updates in one iteration')
    parser.add_argument('--transform', action='store_true', default=False,
                        help='flip or crop images for CNN training')
    parser.add_argument('--lambda_aux_fake', type=float, default=1.0)
    parser.add_argument('--visualize_freq', type=int, default=500, metavar='N')

    ## eval in training
    parser.add_argument('--comp_IS_in_train', action='store_true', default=False)
    parser.add_argument('--comp_IS_freq', type=int, default=1000)

    # DiffAugment setting
    parser.add_argument('--gan_DiffAugment', action='store_true', default=False)
    parser.add_argument('--gan_DiffAugment_policy', type=str, default='color,translation,cutout')

    '''Sampling and Comparing Settings'''
    parser.add_argument('--samp_round', type=int, default=3)
    parser.add_argument('--samp_nfake_per_class', type=int, default=5000)
    parser.add_argument('--samp_batch_size', type=int, default=1000)
    parser.add_argument('--samp_dump_fake_data', action='store_true', default=False)

    parser.add_argument('--inception_from_scratch', action='store_true', default=False)
    parser.add_argument('--eval_real', action='store_true', default=False)
    parser.add_argument('--eval_fake', action='store_true', default=False)
    parser.add_argument('--FID_batch_size', type=int, default=100)

    args = parser.parse_args()

    return args

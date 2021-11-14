import argparse


def gen_synth_data_opts():
    parser = argparse.ArgumentParser()

    ''' Overall settings '''
    parser.add_argument('--root_path', type=str, default='')
    parser.add_argument('--data_path', type=str, default='')
    parser.add_argument('--gan_ckpt_path', type=str, default='')
    parser.add_argument('--embed_ckpt_path', type=str, default='')
    parser.add_argument('--eval_ckpt_path_FID', type=str, default='')
    parser.add_argument('--eval_ckpt_path_LS', type=str, default='', help="Label score")
    parser.add_argument('--eval_ckpt_path_Div', type=str, default='', help="Diversity")
    parser.add_argument('--seed', type=int, default=2021, metavar='S', help='random seed (default: 2020)')
    parser.add_argument('--num_workers', type=int, default=0)

    ''' Datast Settings '''
    parser.add_argument('--data_split', type=str, default='train',
                        choices=['all', 'train'])
    parser.add_argument('--min_label', type=float, default=0.0)
    parser.add_argument('--max_label', type=float, default=90.0)
    parser.add_argument('--num_channels', type=int, default=3, metavar='N')
    parser.add_argument('--img_size', type=int, default=64, metavar='N',
                        choices=[64])
    parser.add_argument('--max_num_img_per_label', type=int, default=25, metavar='N')
    parser.add_argument('--max_num_img_per_label_after_replica', type=int, default=0, metavar='N')
    parser.add_argument('--show_real_imgs', action='store_true', default=False)
    parser.add_argument('--visualize_fake_images', action='store_true', default=False)


    ''' GAN Settings '''
    parser.add_argument('--gan_dim_g', type=int, default=256)
    parser.add_argument('--net_embed', type=str, default='ResNet34_embed') #ResNetXX_emebed
    parser.add_argument('--dim_embed', type=int, default=128) #dimension of the embedding space
    
    ''' Collab setting '''
    parser.add_argument('--disc_shaping', action='store_true', default=False)
    parser.add_argument('--rollout_rate', type=float, default=0.1)
    parser.add_argument('--rollout_steps', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=1000, help='batch size')
    parser.add_argument('--lr_d', type=float, default=1e-4)
    parser.add_argument('--niter', type=int, default=10001, help='number of iterations')
    
    ''' Sampling Settings '''
    parser.add_argument('--subsampling', action='store_true', default=False)
    parser.add_argument('--samp_batch_size', type=int, default=1000) #also used for computing density ratios after the dre training
    parser.add_argument('--samp_burnin_size', type=int, default=1000)
    parser.add_argument('--samp_nfake_per_label', type=int, default=200) #number of fake images per class for evaluation
    parser.add_argument('--samp_dump_fake_data', action='store_true', default=False)

    ''' Evaluation '''
    '''
    Four evaluation modes:
    Mode 1: eval on unique labels used for GAN training;
    Mode 2. eval on all unique labels in the dataset and when computing FID use all real images in the dataset;
    Mode 3. eval on all unique labels in the dataset and when computing FID only use real images for GAN training in the dataset (to test SFID's effectiveness on unseen labels);
    Mode 4. eval on a interval [min_label, max_label] with num_eval_labels labels.
    '''
    parser.add_argument('--eval', action='store_true', default=False)
    parser.add_argument('--eval_mode', type=int, default=2)
    parser.add_argument('--num_eval_labels', type=int, default=-1)
    parser.add_argument('--nreal_per_label', type=int, default=-1)
    parser.add_argument('--FID_radius', type=float, default=0)
    parser.add_argument('--FID_num_centers', type=int, default=-1)
    parser.add_argument('--eval_batch_size', type=int, default=200)
    parser.add_argument('--dump_fake_for_NIQE', action='store_true', default=False)

    args = parser.parse_args()

    return args




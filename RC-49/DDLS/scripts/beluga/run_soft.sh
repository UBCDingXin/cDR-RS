#!/bin/bash
#SBATCH --account=def-wjwelch
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=1-00:00
#SBATCH --mail-user=xin.ding@stat.ubc.ca
#SBATCH --mail-type=ALL
#SBATCH --job-name=RC64_DDLS_S
#SBATCH --output=%x-%j.out


module load arch/avx512 StdEnv/2020
module load cuda/11.0
module load python/3.8.2
virtualenv --no-download ~/ENV
source ~/ENV/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r ./requirements.req


ROOT_PATH="/scratch/dingx92/Subsample_cGANs_via_cDRE/RC-49/RC-49_64x64/DDLS"
DATA_PATH="/scratch/dingx92/datasets/RC-49"
CKPT_GAN_PATH="/scratch/dingx92/Subsample_cGANs_via_cDRE/RC-49/RC-49_64x64/eval_and_gan_ckpts/ckpt_CcGAN_niters_30000_seed_2020_soft_0.04736784919541229_50624.99999999429.pth"
CKPT_EMBED_PATH="/scratch/dingx92/Subsample_cGANs_via_cDRE/RC-49/RC-49_64x64/eval_and_gan_ckpts/ckpt_net_y2h_epoch_500_seed_2020.pth"
CKPT_EVAL_FID_PATH="/scratch/dingx92/Subsample_cGANs_via_cDRE/RC-49/RC-49_64x64/eval_and_gan_ckpts/ckpt_AE_epoch_200_seed_2020_CVMode_False.pth"
CKPT_EVAL_LS_PATH="/scratch/dingx92/Subsample_cGANs_via_cDRE/RC-49/RC-49_64x64/eval_and_gan_ckpts/ckpt_PreCNNForEvalGANs_ResNet34_regre_epoch_200_seed_2020_CVMode_False.pth"
CKPT_EVAL_Div_PATH="/scratch/dingx92/Subsample_cGANs_via_cDRE/RC-49/RC-49_64x64/eval_and_gan_ckpts/ckpt_PreCNNForEvalGANs_ResNet34_class_epoch_200_seed_2020_classify_49_chair_types_CVMode_False.pth"


MIN_LABEL=0.0
MAX_LABEL=90.0
IMG_SIZE=64
MAX_N_IMG_PER_LABEL=25
MAX_N_IMG_PER_LABEL_AFTER_REPLICA=0
SAMP_NFAKE_PER_LABEL=200
SAMP_BS=250
DDLS_N_STEP=200

python main.py \
    --root_path $ROOT_PATH --data_path $DATA_PATH --gan_ckpt_path $CKPT_GAN_PATH --embed_ckpt_path $CKPT_EMBED_PATH \
    --eval_ckpt_path_FID $CKPT_EVAL_FID_PATH --eval_ckpt_path_LS $CKPT_EVAL_LS_PATH --eval_ckpt_path_Div $CKPT_EVAL_Div_PATH \
    --min_label $MIN_LABEL --max_label $MAX_LABEL --img_size $IMG_SIZE \
    --max_num_img_per_label $MAX_N_IMG_PER_LABEL --max_num_img_per_label_after_replica $MAX_N_IMG_PER_LABEL_AFTER_REPLICA \
    --ddls_n_steps $DDLS_N_STEP --ddls_alpha 1 --ddls_step_lr 1e-4 --ddls_eps_std 2e-4 \
    --samp_dump_fake_data \
    --samp_batch_size $SAMP_BS --samp_nfake_per_label $SAMP_NFAKE_PER_LABEL \
    --eval_batch_size 100 --FID_radius 0 \
    2>&1 | tee output_soft_${DDLS_N_STEP}.txt

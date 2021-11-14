#!/bin/bash
#SBATCH --account=def-wjwelch
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=0-08:00
#SBATCH --mail-user=xin.ding@stat.ubc.ca
#SBATCH --mail-type=ALL
#SBATCH --job-name=RC64_DRS_S
#SBATCH --output=%x-%j.out


module load arch/avx512 StdEnv/2020
module load cuda/11.0
module load python/3.8.2
virtualenv --no-download ~/ENV
source ~/ENV/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r ./requirements.req


ROOT_PATH="/scratch/dingx92/Subsample_cGANs_via_cDRE/RC-49/RC-49_64x64/DRS"
DATA_PATH="/scratch/dingx92/datasets/RC-49"
CKPT_GAN_PATH="/scratch/dingx92/Subsample_cGANs_via_cDRE/RC-49/RC-49_64x64/eval_and_gan_ckpts/ckpt_CcGAN_niters_30000_seed_2020_soft_0.04736784919541229_50624.99999999429.pth"
CKPT_EMBED_PATH="/scratch/dingx92/Subsample_cGANs_via_cDRE/RC-49/RC-49_64x64/eval_and_gan_ckpts/ckpt_net_y2h_epoch_500_seed_2020.pth"
CKPT_EVAL_FID_PATH="/scratch/dingx92/Subsample_cGANs_via_cDRE/RC-49/RC-49_64x64/eval_and_gan_ckpts/ckpt_AE_epoch_200_seed_2020_CVMode_False.pth"
CKPT_EVAL_LS_PATH="/scratch/dingx92/Subsample_cGANs_via_cDRE/RC-49/RC-49_64x64/eval_and_gan_ckpts/ckpt_PreCNNForEvalGANs_ResNet34_regre_epoch_200_seed_2020_CVMode_False.pth"
CKPT_EVAL_Div_PATH="/scratch/dingx92/Subsample_cGANs_via_cDRE/RC-49/RC-49_64x64/eval_and_gan_ckpts/ckpt_PreCNNForEvalGANs_ResNet34_class_epoch_200_seed_2020_classify_49_chair_types_CVMode_False.pth"

SEED=2021
NUM_WORKERS=0
MIN_LABEL=0.0
MAX_LABEL=90.0
IMG_SIZE=64
MAX_N_IMG_PER_LABEL=25
MAX_N_IMG_PER_LABEL_AFTER_REPLICA=0

SAMP_BS=600
SAMP_BURNIN=1000
SAMP_NFAKE_PER_LABEL=200

# ###############################
# ## NO keep training
# python main.py \
#     --root_path $ROOT_PATH --data_path $DATA_PATH --gan_ckpt_path $CKPT_GAN_PATH --embed_ckpt_path $CKPT_EMBED_PATH \
#     --eval_ckpt_path_FID $CKPT_EVAL_FID_PATH --eval_ckpt_path_LS $CKPT_EVAL_LS_PATH --eval_ckpt_path_Div $CKPT_EVAL_Div_PATH \
#     --seed $SEED --num_workers $NUM_WORKERS \
#     --min_label $MIN_LABEL --max_label $MAX_LABEL --img_size $IMG_SIZE \
#     --max_num_img_per_label $MAX_N_IMG_PER_LABEL --max_num_img_per_label_after_replica $MAX_N_IMG_PER_LABEL_AFTER_REPLICA \
#     --subsampling --samp_dump_fake_data \
#     --samp_batch_size $SAMP_BS --samp_burnin_size $SAMP_BURNIN --samp_nfake_per_label $SAMP_NFAKE_PER_LABEL \
#     --eval_batch_size 100 --FID_radius 0 \
#     2>&1 | tee output_soft.txt


###############################
## keep training
python main.py \
    --root_path $ROOT_PATH --data_path $DATA_PATH --gan_ckpt_path $CKPT_GAN_PATH --embed_ckpt_path $CKPT_EMBED_PATH \
    --eval_ckpt_path_FID $CKPT_EVAL_FID_PATH --eval_ckpt_path_LS $CKPT_EVAL_LS_PATH --eval_ckpt_path_Div $CKPT_EVAL_Div_PATH \
    --seed $SEED --num_workers $NUM_WORKERS \
    --min_label $MIN_LABEL --max_label $MAX_LABEL --img_size $IMG_SIZE \
    --max_num_img_per_label $MAX_N_IMG_PER_LABEL --max_num_img_per_label_after_replica $MAX_N_IMG_PER_LABEL_AFTER_REPLICA \
    --keep_training --keep_training_niters 2000 --keep_training_batchsize 256 \
    --subsampling --samp_dump_fake_data \
    --samp_batch_size $SAMP_BS --samp_burnin_size $SAMP_BURNIN --samp_nfake_per_label $SAMP_NFAKE_PER_LABEL \
    --eval_batch_size 100 --FID_radius 0 \
    2>&1 | tee output_soft_keepTrain.txt

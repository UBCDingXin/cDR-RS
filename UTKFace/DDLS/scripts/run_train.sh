#!/bin/bash

ROOT_PATH="./UTKFace/DDLS"
DATA_PATH="./datasets/UTKFace"
CKPT_GAN_PATH="./UTKFace/eval_and_gan_ckpts/ckpt_CcGAN_niters_40000_seed_2020_soft_0.04092845142095955_3599.9999999999777.pth"
CKPT_EMBED_PATH="./UTKFace/eval_and_gan_ckpts/ckpt_net_y2h_epoch_500_seed_2020.pth"
CKPT_EVAL_FID_PATH="./UTKFace/eval_and_gan_ckpts/ckpt_AE_epoch_200_seed_2020_CVMode_False.pth"
CKPT_EVAL_LS_PATH="./UTKFace/eval_and_gan_ckpts/ckpt_PreCNNForEvalGANs_ResNet34_regre_epoch_200_seed_2020_CVMode_False.pth"
CKPT_EVAL_Div_PATH="./UTKFace/eval_and_gan_ckpts/ckpt_PreCNNForEvalGANs_ResNet34_class_epoch_200_seed_2020_classify_5_races_CVMode_False.pth"

MIN_LABEL=1
MAX_LABEL=60
IMG_SIZE=64
MAX_N_IMG_PER_LABEL=999999
MAX_N_IMG_PER_LABEL_AFTER_REPLICA=0
SAMP_NFAKE_PER_LABEL=1000
SAMP_BS=250
DDLS_N_STEP=1000

python main.py \
    --root_path $ROOT_PATH --data_path $DATA_PATH --gan_ckpt_path $CKPT_GAN_PATH --embed_ckpt_path $CKPT_EMBED_PATH \
    --eval_ckpt_path_FID $CKPT_EVAL_FID_PATH --eval_ckpt_path_LS $CKPT_EVAL_LS_PATH --eval_ckpt_path_Div $CKPT_EVAL_Div_PATH \
    --min_label $MIN_LABEL --max_label $MAX_LABEL --img_size $IMG_SIZE \
    --max_num_img_per_label $MAX_N_IMG_PER_LABEL --max_num_img_per_label_after_replica $MAX_N_IMG_PER_LABEL_AFTER_REPLICA \
    --ddls_n_steps $DDLS_N_STEP --ddls_alpha 1 --ddls_step_lr 1e-4 --ddls_eps_std 2e-4 \
    --samp_dump_fake_data \
    --samp_batch_size $SAMP_BS --samp_nfake_per_label $SAMP_NFAKE_PER_LABEL \
    --eval --eval_batch_size 100 --FID_radius 0 \
    2>&1 | tee output_DDLS_${DDLS_N_STEP}.txt

#!/bin/bash

ROOT_PATH="./RC-49/DRS"
DATA_PATH="./datasets/RC-49"
CKPT_GAN_PATH="./RC-49/eval_and_gan_ckpts/ckpt_CcGAN_niters_30000_seed_2020_soft_0.04736784919541229_50624.99999999429.pth"
CKPT_EMBED_PATH="./RC-49/eval_and_gan_ckpts/ckpt_net_y2h_epoch_500_seed_2020.pth"
CKPT_EVAL_FID_PATH="./RC-49/eval_and_gan_ckpts/ckpt_AE_epoch_200_seed_2020_CVMode_False.pth"
CKPT_EVAL_LS_PATH="./RC-49/eval_and_gan_ckpts/ckpt_PreCNNForEvalGANs_ResNet34_regre_epoch_200_seed_2020_CVMode_False.pth"
CKPT_EVAL_Div_PATH="./RC-49/eval_and_gan_ckpts/ckpt_PreCNNForEvalGANs_ResNet34_class_epoch_200_seed_2020_classify_49_chair_types_CVMode_False.pth"

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

python main.py \
    --root_path $ROOT_PATH --data_path $DATA_PATH --gan_ckpt_path $CKPT_GAN_PATH --embed_ckpt_path $CKPT_EMBED_PATH \
    --eval_ckpt_path_FID $CKPT_EVAL_FID_PATH --eval_ckpt_path_LS $CKPT_EVAL_LS_PATH --eval_ckpt_path_Div $CKPT_EVAL_Div_PATH \
    --seed $SEED --num_workers $NUM_WORKERS \
    --min_label $MIN_LABEL --max_label $MAX_LABEL --img_size $IMG_SIZE \
    --max_num_img_per_label $MAX_N_IMG_PER_LABEL --max_num_img_per_label_after_replica $MAX_N_IMG_PER_LABEL_AFTER_REPLICA \
    --keep_training --keep_training_niters 1000 --keep_training_batchsize 128 \
    --subsampling --samp_dump_fake_data \
    --samp_batch_size $SAMP_BS --samp_burnin_size $SAMP_BURNIN --samp_nfake_per_label $SAMP_NFAKE_PER_LABEL \
    --eval --eval_batch_size 100 --FID_radius 0 \
    2>&1 | tee output_DRS_keepTrain.txt

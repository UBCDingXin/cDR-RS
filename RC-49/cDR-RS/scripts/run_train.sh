#!/bin/bash

ROOT_PATH="./RC-49/cDR-RS"
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

DRE_PRESAE_EPOCHS=200
DRE_PRESAE_BS=128
DRE_PRESAE_SPARSITY=0.001
DRE_PRESAE_REGRESSION=1

DRE_DR="MLP5"
DRE_DR_EPOCHS=200
DRE_DR_LR_BASE=1e-4
DRE_DR_BS=256
OPTIMIZER="ADAM"
DRE_DR_LAMBDA=0.01
DRE_ADJUST_NITERS=200

SAMP_BS=600
SAMP_BURNIN=1000
SAMP_NFAKE_PER_LABEL=200


DRE_KAPPA=-6
python main.py \
    --root_path $ROOT_PATH --data_path $DATA_PATH --gan_ckpt_path $CKPT_GAN_PATH --embed_ckpt_path $CKPT_EMBED_PATH \
    --eval_ckpt_path_FID $CKPT_EVAL_FID_PATH --eval_ckpt_path_LS $CKPT_EVAL_LS_PATH --eval_ckpt_path_Div $CKPT_EVAL_Div_PATH \
    --seed $SEED --num_workers $NUM_WORKERS \
    --min_label $MIN_LABEL --max_label $MAX_LABEL --img_size $IMG_SIZE \
    --max_num_img_per_label $MAX_N_IMG_PER_LABEL --max_num_img_per_label_after_replica $MAX_N_IMG_PER_LABEL_AFTER_REPLICA \
    --dre_presae_epochs $DRE_PRESAE_EPOCHS --dre_presae_resume_epoch 0 \
    --dre_presae_lr_base 0.01 --dre_presae_lr_decay_factor 0.1 --dre_presae_lr_decay_freq 50 \
    --dre_presae_batch_size_train $DRE_PRESAE_BS --dre_presae_weight_decay 1e-4 \
    --dre_presae_lambda_sparsity $DRE_PRESAE_SPARSITY --dre_presae_lambda_regression $DRE_PRESAE_REGRESSION \
    --dre_net $DRE_DR --dre_epochs $DRE_DR_EPOCHS --dre_resume_epoch 0 --dre_save_freq 50 \
    --dre_lr_base $DRE_DR_LR_BASE --dre_lr_decay_epochs 80_150 --dre_optimizer $OPTIMIZER \
    --dre_batch_size $DRE_DR_BS --dre_lambda $DRE_DR_LAMBDA \
    --dre_kappa $DRE_KAPPA --dre_adjust_niters $DRE_ADJUST_NITERS \
    --subsampling --samp_dump_fake_data \
    --samp_batch_size $SAMP_BS --samp_burnin_size $SAMP_BURNIN --samp_nfake_per_label $SAMP_NFAKE_PER_LABEL \
    --eval --eval_batch_size 100 --FID_radius 0 \
    2>&1 | tee output_soft_cDR-RS_DR_${DRE_DR}_${DRE_DR_LAMBDA}_${DRE_KAPPA}_presae_${DRE_PRESAE_SPARSITY}_${DRE_PRESAE_REGRESSION}.txt
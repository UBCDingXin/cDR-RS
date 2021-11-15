#!/bin/bash

ROOT_PATH="./CIFAR-10/GOLD"
DATA_PATH="./datasets/CIFAR-10"
EVAL_PATH="./CIFAR-10/eval_and_gan_ckpts/ckpt_PreCNNForEval_InceptionV3_epoch_200_SEED_2021_Transformation_True.pth"
GAN_CKPT_PATH="./CIFAR-10/eval_and_gan_ckpts/ckpt_ACGAN_niters_100000_nDs_1_seed_2021.pth"

SEED=2021
GAN_NET="ACGAN"
SAMP_NROUNDS=1
SAMP_BS=500
SAMP_BURNIN=50000
SAMP_NFAKE_PER_CLASS=10000

python main.py \
    --root_path $ROOT_PATH --data_path $DATA_PATH --eval_ckpt_path $EVAL_PATH --seed $SEED \
    --gan_net $GAN_NET --gan_ckpt_path $GAN_CKPT_PATH \
    --drs_perc 50 \
    --samp_round $SAMP_NROUNDS --samp_batch_size $SAMP_BS --samp_burnin_size $SAMP_BURNIN \
    --samp_nfake_per_class $SAMP_NFAKE_PER_CLASS --samp_dump_fake_data \
    --inception_from_scratch --eval --eval_FID_batch_size 200 \
    2>&1 | tee output.txt

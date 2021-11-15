#!/bin/bash

ROOT_PATH="./CIFAR-10/GANs/ACGAN"
DATA_PATH="./datasets/CIFAR-10"
EVAL_CKPT_PATH="./CIFAR-10/eval_and_gan_ckpts/ckpt_PreCNNForEval_InceptionV3_epoch_200_SEED_2021_Transformation_True.pth"

SEED=2021
NITERS=100000
BATCHSIZE=512
LR_G=2e-4
LR_D=2e-4
nDs=1
SAVE_FREQ=5000
VISUAL_FREQ=1000
COMP_IS_FREQ=2000
NFAKE_PER_CLASS=10000
SAMP_ROUND=1

resume_niter=0
python main.py \
    --root_path $ROOT_PATH --data_path $DATA_PATH --eval_ckpt_path $EVAL_CKPT_PATH --seed $SEED \
    --gan_arch ACGAN --niters $NITERS --resume_niter $resume_niter --save_freq $SAVE_FREQ --visualize_freq $VISUAL_FREQ \
    --batch_size $BATCHSIZE --lr_g $LR_G --lr_d $LR_D --num_D_steps $nDs --lambda_aux_fake 0.1 \
    --transform \
    --comp_IS_in_train --comp_IS_freq $COMP_IS_FREQ \
    --samp_round $SAMP_ROUND --samp_nfake_per_class $NFAKE_PER_CLASS --samp_batch_size 1000 --samp_dump_fake_data \
    --inception_from_scratch --eval_fake --FID_batch_size 200 \
    2>&1 | tee output_ACGAN_training.txt
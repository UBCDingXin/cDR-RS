@echo off

set ROOT_PATH="G:/OneDrive/Working_directory/Subsample_cGANs_via_cDRE/CIFAR-10/GANs/SNGAN"
set DATA_PATH="G:/OneDrive/Working_directory/datasets/CIFAR-10"
set EVAL_CKPT_PATH="G:/OneDrive/Working_directory/Subsample_cGANs_via_cDRE/CIFAR-10/eval_models/ckpt_PreCNNForEval_InceptionV3_epoch_200_SEED_2021_Transformation_True.pth"

set SEED=2021
set NITERS=50000
set BATCHSIZE=256
set LR_G=1e-4
set LR_D=1e-4
set nDs=4
set SAVE_FREQ=5000
set LOSS_TYPE="hinge"
set VISUAL_FREQ=1000
set COMP_IS_FREQ=2000
set NFAKE_PER_CLASS=10000
set SAMP_ROUND=1


python main.py ^
    --root_path %ROOT_PATH% --data_path %DATA_PATH% --eval_ckpt_path %EVAL_CKPT_PATH% --seed %SEED% ^
    --gan_arch SNGAN --niters %NITERS% --resume_niter 0 --save_freq %SAVE_FREQ% --visualize_freq 1000 ^
    --batch_size %BATCHSIZE% --lr_g %LR_G% --lr_d %LR_D% --num_D_steps %nDs% --loss_type_gan %LOSS_TYPE% --visualize_freq %VISUAL_FREQ% ^
    --transform ^
    --comp_IS_in_train --comp_IS_freq %COMP_IS_FREQ% ^
    --samp_round %SAMP_ROUND% --samp_nfake_per_class %NFAKE_PER_CLASS% --samp_batch_size 1000 --samp_dump_fake_data ^
    --inception_from_scratch --eval_fake --FID_batch_size 200 ^ %*


@REM python main.py ^
@REM     --root_path %ROOT_PATH% --data_path %DATA_PATH% --eval_ckpt_path %EVAL_CKPT_PATH% --seed %SEED% ^
@REM     --gan_arch SNGAN --niters %NITERS% --resume_niter 0 --save_freq %SAVE_FREQ% --visualize_freq 1000 ^
@REM     --batch_size %BATCHSIZE% --lr_g %LR_G% --lr_d %LR_D% --num_D_steps %nDs% --loss_type_gan %LOSS_TYPE% --visualize_freq %VISUAL_FREQ% ^
@REM     --transform ^
@REM     --samp_round %SAMP_ROUND% --samp_nfake_per_class %NFAKE_PER_CLASS% --samp_batch_size 1000 --samp_dump_fake_data ^
@REM     --inception_from_scratch --eval_real --inception_from_scratch --FID_batch_size 200 ^ %*
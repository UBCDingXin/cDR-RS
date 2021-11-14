::===============================================================
:: This is a batch script for running the program on windows 10! 
::===============================================================

@echo off

set ROOT_PATH="G:/OneDrive/Working_directory/Subsample_cGANs_via_cDRE/RC-49/RC-49_64x64/DRS"
set DATA_PATH="G:/OneDrive/Working_directory/datasets/RC-49"
set CKPT_GAN_PATH="G:/OneDrive/Working_directory/Subsample_cGANs_via_cDRE/RC-49/RC-49_64x64/eval_and_gan_ckpts/ckpt_CcGAN_niters_30000_seed_2020_soft_0.04736784919541229_50624.99999999429.pth"
set CKPT_EMBED_PATH="G:/OneDrive/Working_directory/Subsample_cGANs_via_cDRE/RC-49/RC-49_64x64/eval_and_gan_ckpts/ckpt_net_y2h_epoch_500_seed_2020.pth"
set CKPT_EVAL_FID_PATH="G:/OneDrive/Working_directory/Subsample_cGANs_via_cDRE/RC-49/RC-49_64x64/eval_and_gan_ckpts/ckpt_AE_epoch_200_seed_2020_CVMode_False.pth"
set CKPT_EVAL_LS_PATH="G:/OneDrive/Working_directory/Subsample_cGANs_via_cDRE/RC-49/RC-49_64x64/eval_and_gan_ckpts/ckpt_PreCNNForEvalGANs_ResNet34_regre_epoch_200_seed_2020_CVMode_False.pth"
set CKPT_EVAL_Div_PATH="G:/OneDrive/Working_directory/Subsample_cGANs_via_cDRE/RC-49/RC-49_64x64/eval_and_gan_ckpts/ckpt_PreCNNForEvalGANs_ResNet34_class_epoch_200_seed_2020_classify_49_chair_types_CVMode_False.pth"


set SEED=2021
set NUM_WORKERS=0
set MIN_LABEL=0.0
set MAX_LABEL=90.0
set IMG_SIZE=64
set MAX_N_IMG_PER_LABEL=25
set MAX_N_IMG_PER_LABEL_AFTER_REPLICA=0

set SAMP_BS=600
set SAMP_BURNIN=1000
set SAMP_NFAKE_PER_LABEL=50


python main.py ^
    --root_path %ROOT_PATH% --data_path %DATA_PATH% --gan_ckpt_path %CKPT_GAN_PATH% --embed_ckpt_path %CKPT_EMBED_PATH% ^
    --eval_ckpt_path_FID %CKPT_EVAL_FID_PATH% --eval_ckpt_path_LS %CKPT_EVAL_LS_PATH% --eval_ckpt_path_Div %CKPT_EVAL_Div_PATH% ^
    --seed %SEED% --num_workers %NUM_WORKERS% ^
    --min_label %MIN_LABEL% --max_label %MAX_LABEL% --img_size %IMG_SIZE% ^
    --max_num_img_per_label %MAX_N_IMG_PER_LABEL% --max_num_img_per_label_after_replica %MAX_N_IMG_PER_LABEL_AFTER_REPLICA% ^
    --subsampling --samp_dump_fake_data ^
    --samp_batch_size %SAMP_BS% --samp_burnin_size %SAMP_BURNIN% --samp_nfake_per_label %SAMP_NFAKE_PER_LABEL% ^
    --eval --eval_batch_size 100 --FID_radius 0 ^ %*


@REM python main.py ^
@REM     --root_path %ROOT_PATH% --data_path %DATA_PATH% --gan_ckpt_path %CKPT_GAN_PATH% --embed_ckpt_path %CKPT_EMBED_PATH% ^
@REM     --eval_ckpt_path_FID %CKPT_EVAL_FID_PATH% --eval_ckpt_path_LS %CKPT_EVAL_LS_PATH% --eval_ckpt_path_Div %CKPT_EVAL_Div_PATH% ^
@REM     --seed %SEED% --num_workers %NUM_WORKERS% ^
@REM     --min_label %MIN_LABEL% --max_label %MAX_LABEL% --img_size %IMG_SIZE% ^
@REM     --max_num_img_per_label %MAX_N_IMG_PER_LABEL% --max_num_img_per_label_after_replica %MAX_N_IMG_PER_LABEL_AFTER_REPLICA% ^
@REM     --keep_training --keep_training_niters 1000 --keep_training_batchsize 128 ^
@REM     --subsampling --samp_dump_fake_data ^
@REM     --samp_batch_size %SAMP_BS% --samp_burnin_size %SAMP_BURNIN% --samp_nfake_per_label %SAMP_NFAKE_PER_LABEL% ^
@REM     --eval --eval_batch_size 100 --FID_radius 0 ^ %*
::===============================================================
:: This is a batch script for running the program on windows 10! 
::===============================================================

@echo off

set ROOT_PATH="G:/OneDrive/Working_directory/Subsample_cGANs_via_cDRE/RC-49/RC-49_64x64/DDLS"
set DATA_PATH="G:/OneDrive/Working_directory/datasets/RC-49"
set CKPT_GAN_PATH="G:/OneDrive/Working_directory/Subsample_cGANs_via_cDRE/RC-49/RC-49_64x64/eval_and_gan_ckpts/ckpt_CcGAN_niters_30000_seed_2020_hard_0.04736784919541229_0.004444444444444695.pth"
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

set SAMP_BS=200
set SAMP_NFAKE_PER_LABEL=200


python main.py ^
    --root_path %ROOT_PATH% --data_path %DATA_PATH% --gan_ckpt_path %CKPT_GAN_PATH% --embed_ckpt_path %CKPT_EMBED_PATH% ^
    --eval_ckpt_path_FID %CKPT_EVAL_FID_PATH% --eval_ckpt_path_LS %CKPT_EVAL_LS_PATH% --eval_ckpt_path_Div %CKPT_EVAL_Div_PATH% ^
    --seed %SEED% --num_workers %NUM_WORKERS% ^
    --min_label %MIN_LABEL% --max_label %MAX_LABEL% --img_size %IMG_SIZE% ^
    --max_num_img_per_label %MAX_N_IMG_PER_LABEL% --max_num_img_per_label_after_replica %MAX_N_IMG_PER_LABEL_AFTER_REPLICA% ^
    --ddls_n_steps 1000 --ddls_alpha 1 --ddls_step_lr 1e-4 --ddls_eps_std 2e-4 ^
    --samp_dump_fake_data ^
    --samp_batch_size %SAMP_BS% --samp_nfake_per_label %SAMP_NFAKE_PER_LABEL% ^
    --eval --eval_batch_size 100 --FID_radius 0 ^ %*
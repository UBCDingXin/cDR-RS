::===============================================================
:: This is a batch script for running the program on windows 10! 
::===============================================================

@echo off

set ROOT_PATH="./UTKFace/DRS"
set DATA_PATH="./datasets/UTKFace"
set CKPT_GAN_PATH="./UTKFace/eval_and_gan_ckpts/ckpt_CcGAN_niters_40000_seed_2020_soft_0.04092845142095955_3599.9999999999777.pth"
set CKPT_EMBED_PATH="./UTKFace/eval_and_gan_ckpts/ckpt_net_y2h_epoch_500_seed_2020.pth"
set CKPT_EVAL_FID_PATH="./UTKFace/eval_and_gan_ckpts/ckpt_AE_epoch_200_seed_2020_CVMode_False.pth"
set CKPT_EVAL_LS_PATH="./UTKFace/eval_and_gan_ckpts/ckpt_PreCNNForEvalGANs_ResNet34_regre_epoch_200_seed_2020_CVMode_False.pth"
set CKPT_EVAL_Div_PATH="./UTKFace/eval_and_gan_ckpts/ckpt_PreCNNForEvalGANs_ResNet34_class_epoch_200_seed_2020_classify_5_races_CVMode_False.pth"


set SEED=2021
set NUM_WORKERS=0
set MIN_LABEL=1
set MAX_LABEL=60
set IMG_SIZE=64
set MAX_N_IMG_PER_LABEL=999999
set MAX_N_IMG_PER_LABEL_AFTER_REPLICA=0
set SAMP_BS=600
set SAMP_BURNIN=5000
set SAMP_NFAKE_PER_LABEL=1000


python main.py ^
    --root_path %ROOT_PATH% --data_path %DATA_PATH% --gan_ckpt_path %CKPT_GAN_PATH% --embed_ckpt_path %CKPT_EMBED_PATH% ^
    --eval_ckpt_path_FID %CKPT_EVAL_FID_PATH% --eval_ckpt_path_LS %CKPT_EVAL_LS_PATH% --eval_ckpt_path_Div %CKPT_EVAL_Div_PATH% ^
    --seed %SEED% --num_workers %NUM_WORKERS% ^
    --min_label %MIN_LABEL% --max_label %MAX_LABEL% --img_size %IMG_SIZE% ^
    --max_num_img_per_label %MAX_N_IMG_PER_LABEL% --max_num_img_per_label_after_replica %MAX_N_IMG_PER_LABEL_AFTER_REPLICA% ^
    --keep_training --keep_training_niters 2000 --keep_training_batchsize 128 ^
    --subsampling --samp_dump_fake_data ^
    --samp_batch_size %SAMP_BS% --samp_burnin_size %SAMP_BURNIN% --samp_nfake_per_label %SAMP_NFAKE_PER_LABEL% ^
    --eval --eval_batch_size 100 --FID_radius 0 --dump_fake_for_NIQE ^ %*
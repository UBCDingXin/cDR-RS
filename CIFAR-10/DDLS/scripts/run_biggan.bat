@echo off

set ROOT_PATH="./CIFAR-10/DDLS"
set DATA_PATH="./datasets/CIFAR-10"
set EVAL_PATH="./CIFAR-10/eval_and_gan_ckpts/ckpt_PreCNNForEval_InceptionV3_epoch_200_SEED_2021_Transformation_True.pth"
set GAN_G_CKPT_PATH="./CIFAR-10/eval_and_gan_ckpts/BigGAN_39K/G_ema.pth"
set GAN_D_CKPT_PATH="./CIFAR-10/eval_and_gan_ckpts/BigGAN_39K/D.pth"

set SEED=2021
set GAN_NET="BigGAN"
set SAMP_NROUNDS=1
set SAMP_BS=500
set SAMP_BURNIN=5000
set SAMP_NFAKE_PER_CLASS=10000

python main.py ^
    --root_path %ROOT_PATH% --data_path %DATA_PATH% --eval_ckpt_path %EVAL_PATH% --seed %SEED% ^
    --gan_net %GAN_NET% --gan_gene_ckpt_path %GAN_G_CKPT_PATH% --gan_disc_ckpt_path %GAN_D_CKPT_PATH% ^
    --ddls_n_steps 1000 --ddls_alpha 1 --ddls_step_lr 1e-4 --ddls_eps_std 2e-4 ^
    --samp_round %SAMP_NROUNDS% --samp_batch_size %SAMP_BS% --samp_burnin_size %SAMP_BURNIN% ^
    --samp_nfake_per_class %SAMP_NFAKE_PER_CLASS% --samp_dump_fake_data ^
    --inception_from_scratch --eval --eval_FID_batch_size 200 ^ %*
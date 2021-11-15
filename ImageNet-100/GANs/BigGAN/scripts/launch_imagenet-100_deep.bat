@echo off

set ROOT_PATH="./ImageNet-100/GANs/BigGAN"
set DATA_PATH="./datasets/ImageNet-100/"

set EPOCHS=2000
set BATCHSIZE=128

python train.py ^
--model BigGANdeep ^
--root_path %ROOT_PATH% --seed 2021 ^
--data_root %DATA_PATH% --dataset I100_128_hdf5 --parallel --shuffle --num_workers 0 --batch_size %BATCHSIZE% --load_in_mem --no_pin_memory --augment ^
--num_epochs %EPOCHS% ^
--num_G_accumulations 8 --num_D_accumulations 8 ^
--num_D_steps 1 --G_lr 1e-4 --D_lr 4e-4 --D_B2 0.999 --G_B2 0.999 ^
--G_attn 64 --D_attn 64 ^
--G_ch 128 --D_ch 128 ^
--G_depth 2 --D_depth 2 ^
--G_nl inplace_relu --D_nl inplace_relu ^
--SN_eps 1e-6 --BN_eps 1e-5 --adam_eps 1e-6 ^
--G_ortho 0.0 ^
--G_shared ^
--G_init ortho --D_init ortho ^
--hier --dim_z 128 --shared_dim 128 ^
--ema --use_ema --ema_start 20000 --G_eval_mode ^
--test_every 1000 --no_fid --save_every 1000 --num_best_copies 5 --num_save_copies 2 ^
--DiffAugment_policy "color,translation,cutout" ^ %*
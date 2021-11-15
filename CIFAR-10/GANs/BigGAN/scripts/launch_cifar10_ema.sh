#!/bin/bash
#SBATCH --account=def-wjwelch
#SBATCH --gres=gpu:v100l:1
#SBATCH --cpus-per-task=3
#SBATCH --mem=64G
#SBATCH --time=1-00:00
#SBATCH --mail-user=xin.ding@stat.ubc.ca
#SBATCH --mail-type=ALL
#SBATCH --job-name=C10_BG
#SBATCH --output=%x-%j.out


module load arch/avx512 StdEnv/2020
module load cuda/11.0
module load python/3.8.2
virtualenv --no-download ~/ENV
source ~/ENV/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r ./requirements.req



ROOT_PATH="/scratch/dingx92/Subsample_cGANs_via_cDRE/CIFAR-10/GANs/BigGAN"

EPOCHS=2000
BATCHSIZE=512

### complete CIFAR10 dataset
python train.py \
--root_path $ROOT_PATH --seed 2021 \
--shuffle --batch_size $BATCHSIZE --parallel --num_workers 0  --no_pin_memory \
--num_G_accumulations 1 --num_D_accumulations 1 --num_epochs $EPOCHS \
--num_D_steps 4 --G_lr 2e-4 --D_lr 2e-4 \
--data_root data/ --dataset C10 --augment \
--G_ortho 0.0 \
--G_attn 0 --D_attn 0 \
--G_init N02 --D_init N02 \
--ema --use_ema --ema_start 1000 \
--test_every 1000 --no_fid --save_every 1000 --num_best_copies 5 --num_save_copies 2 --seed 0 \
2>&1 | tee output_biggan_cifar10_full_2020.txt
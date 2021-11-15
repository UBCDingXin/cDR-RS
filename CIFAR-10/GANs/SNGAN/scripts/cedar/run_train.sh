#!/bin/bash
#SBATCH --account=def-wjwelch
#SBATCH --gres=gpu:v100l:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=1-00:00
#SBATCH --mail-user=xin.ding@stat.ubc.ca
#SBATCH --mail-type=ALL
#SBATCH --job-name=CI10_SNGAN
#SBATCH --output=%x-%j.out


module load arch/avx512 StdEnv/2020
module load cuda/11.0
module load python/3.8.2
virtualenv --no-download ~/ENV
source ~/ENV/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r ./requirements.req


ROOT_PATH="/scratch/dingx92/Subsample_cGANs_via_cDRE/CIFAR-10/GANs/SNGAN"
DATA_PATH="/scratch/dingx92/datasets/CIFAR-10"
EVAL_CKPT_PATH="/scratch/dingx92/Subsample_cGANs_via_cDRE/CIFAR-10/eval_models/ckpt_PreCNNForEval_InceptionV3_epoch_200_SEED_2021_Transformation_True.pth"

SEED=2021
NITERS=100000
BATCHSIZE=512
LR_G=1e-4
LR_D=2e-4
nDs=1
SAVE_FREQ=5000
LOSS_TYPE="hinge"
VISUAL_FREQ=1000
COMP_IS_FREQ=2000
NFAKE_PER_CLASS=10000
SAMP_ROUND=1

python main.py \
    --root_path $ROOT_PATH --data_path $DATA_PATH --eval_ckpt_path $EVAL_CKPT_PATH --seed $SEED \
    --gan_arch SNGAN --niters $NITERS --resume_niter 0 --save_freq $SAVE_FREQ --visualize_freq 1000 \
    --batch_size $BATCHSIZE --lr_g $LR_G --lr_d $LR_D --num_D_steps $nDs --loss_type_gan $LOSS_TYPE --visualize_freq $VISUAL_FREQ \
    --transform \
    --comp_IS_in_train --comp_IS_freq $COMP_IS_FREQ \
    --samp_round $SAMP_ROUND --samp_nfake_per_class $NFAKE_PER_CLASS --samp_batch_size 1000 --samp_dump_fake_data \
    --inception_from_scratch --eval_fake --FID_batch_size 500 \
    2>&1 | tee output_SNGAN_training.txt
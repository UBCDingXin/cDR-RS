# Efficient Density Ratio-Guided Subsampling of Conditional GANs, With Conditioning on a Class or a Continuous Variable

--------------------------------------------------------

This repository provides the source codes for the experiments in our paper at [https://arxiv.org/abs/2103.11166v2](https://arxiv.org/abs/2103.11166v2). <br />
If you use this code, please cite
```text
@misc{ding2021efficient,
      title={Efficient Density Ratio-Guided Subsampling of Conditional GANs, With Conditioning on a Class or a Continuous Variable}, 
      author={Xin Ding and Yongwei Wang and Z. Jane Wang and William J. Welch},
      year={2021},
      eprint={2103.11166v2},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

--------------------------------------------------------

# Repository Structure

```
├── CIFAR-10
│   ├── cDR-RS
│   ├── DRE-F-SP+RS
│   ├── DRS
│   ├── Collab
│   ├── DDLS
│   ├── GOLD
│   ├── GANs
│   └── eval_and_gan_ckpts
│
├── CIFAR-100
│   ├── cDR-RS
│   ├── DRE-F-SP+RS
│   ├── DRS
│   ├── Collab
│   ├── DDLS
│   ├── GANs
│   └── eval_and_gan_ckpts
│
├── ImageNet-100
│   ├── cDR-RS
│   ├── DRE-F-SP+RS
│   ├── DRS
│   ├── Collab
│   ├── DDLS
│   ├── GANs
│   └── eval_and_gan_ckpts
│
├── UTKFace
│   ├── cDR-RS
│   ├── DRE-F-SP+RS
│   ├── DRS
│   ├── Collab
│   ├── DDLS
│   └── eval_and_gan_ckpts
│
└── RC-49
    ├── cDR-RS
    ├── DRS
    ├── Collab
    └── eval_and_gan_ckpts

```

--------------------------------------------------------

# The overall workflow of cDR-RS

<p align="center">
  <img src="images/workflow_cDR-RS.png">
  The overall workflow of cDR-RS.
</p>

--------------------------------------------------------

# Effectiveness and Efficiency Comparison on ImageNet-100 and UTKFace

<!-- ImageNet-100                  |  UTKFace
:-------------------------:|:-------------------------:
![](images/ImageNet-100_BigGANdeep_efficiency_and_effectiveness_analysis_3bars.png)  |  ![](images/UTKFace_64x64_SVDL_efficiency_and_effectiveness_analysis_3bars_LS_kappa-6.png) -->


<p align="center">
  <img src="images/ImageNet-100_BigGANdeep_efficiency_and_effectiveness_analysis_3bars.png">
  Effectiveness and Efficiency Comparison on ImageNet-100 (Two NVIDIA V100)
</p>

<p align="center">
  <img src="images/UTKFace_64x64_SVDL_efficiency_and_effectiveness_analysis_3bars_LS_kappa-6.png">
  Effectiveness and Efficiency Comparison on UTKFace (One NVIDIA V100)
</p>

-------------------------------

# Software Requirements
| Item | Version |
|---|---|
| Python|3.9.5|
| argparse | 1.1 |
| CUDA  | 11.4 |
| cuDNN| 8.2|
| numpy | 1.14 |
| torch | 1.9.0 |
| torchvision | 0.10.0 |
| Pillow | 8.2.0 |
| matplotlib | 3.4.2 |
| tqdm | 4.61.1 |
| h5py | 3.3.0 |
| Matlab | 2020a |


--------------------------------------------------------

# Datasets

The unprocessed ImageNet-100 dataset (`imagenet.tar.gz`) can be download from [here](https://drive.google.com/drive/folders/0B7IzDz-4yH_HOXdoaDU4dk40RFE?usp=sharing). <br />
After unzipping `imagenet.tar.gz`, put `image` in `./datasets/ImageNet-100`. Then run `python make_dataset.py` in `./datasets/ImageNet-100`. Finally, we will get the h5 file of the processed ImageNet-100 dataset named `ImageNet_128x128_100Class.h5`. <br /> 

Please refer to [https://github.com/UBCDingXin/improved_CcGAN](https://github.com/UBCDingXin/improved_CcGAN) for the download link of RC-49 and the preprocessed UTKFace datasets. Download RC-49 (64x64) and UTKFace (64x64) h5 files and put them in `./datasets/RC-49` and `./datasets/UTKFace`, respectively. <br />


--------------------------------------------------------

# Sample Usage

**Remember to set correct root path, data path, and checkpoint path. Please also remember to download necessary checkpoints for each experiment.** <br />

## 1. Sampling From Class-conditional GANs

### CIFAR-10 (`./CIFAR-10`)
Download [eval_and_gan_ckpts.zip](https://1drv.ms/u/s!Arj2pETbYnWQuZpeWtw5xejKpnXcsg?e=f7zSVO). Unzip `eval_and_gan_ckpts.zip` to get `eval_and_gan_ckpts`, and move `eval_and_gan_ckpts` to `./CIFAR-10`. This folder includes the checkpoint of Inception-V3 for evaluation.  <br />

1. **Train three GANs**: ACGAN, SNGAN, and BigGAN. Their checkpoints used in our experiment are also provided in `eval_and_gan_ckpts`. Thus, to reproduce our results, the training of these GANs is actually not necessary. <br />
ACGAN: Run `./CIFAR-10/GANs/ACGAN/scripts/run_train.sh` <br />
SNGAN: Run `./CIFAR-10/GANs/SNGAN/scripts/run_train.sh` <br />
BigGAN: Run `./CIFAR-10/GANs/BigGAN/scripts/launch_cifar10_ema.sh` <br />
2. **Implement each sampling method.** Run `.sh` script(s) in the folder of each method. <br />
**cDR-RS and DRE-F-SP+RS**: Run `./scripts/run_exp_acgan.sh` for ACGAN. Run `./scripts/run_exp_sngan.sh` for SNGAN. Run `./scripts/run_exp_biggan.sh` for BigGAN. <br />
**DRS, DDLS, and Collab**: Run `./scripts/run_sngan.sh` for SNGAN. Run `./scripts/run_biggan.sh` for BigGAN. <br />
**GOLD**: Run `./scripts/run_acgan.sh` for ACGAN. <br />


### CIFAR-100 (`./CIFAR-100`)
Download [eval_and_gan_ckpts.zip](https://1drv.ms/u/s!Arj2pETbYnWQuZp-cl2SSkwVVz-VVA?e=z9bNnw). Unzip `eval_and_gan_ckpts.zip` to get `eval_and_gan_ckpts`, and move `eval_and_gan_ckpts` to `./CIFAR-100`. This folder includes the checkpoint of Inception-V3 for evaluation.  <br />

1. **Train BigGAN.** Its checkpoints used in our experiment are also provided in `eval_and_gan_ckpts`. Thus, to reproduce our results, the training of BigGAN is actually not necessary. <br />
BigGAN: Run `./CIFAR-100/GANs/BigGAN/scripts/launch_cifar100_ema.sh` <br />
2. **Implement each sampling method.** Run `.sh` script(s) in the folder of each method.  <br />
**cDR-RS and DRE-F-SP+RS**: Run `./scripts/run_exp_biggan.sh` for BigGAN. <br />
**DRS, DDLS, and Collab**: Run `./scripts/run_biggan.sh` for BigGAN. <br />


### ImageNet-100 (`./ImageNet-100`)
Download [eval_and_gan_ckpts.zip](https://1drv.ms/u/s!Arj2pETbYnWQuZp9jGr2qkLwQ-TYbw?e=IS5zL4). Unzip `eval_and_gan_ckpts.zip` to get `eval_and_gan_ckpts`, and move `eval_and_gan_ckpts` to `./ImageNet-100`. This folder includes the checkpoint of Inception-V3 for evaluation.  <br />

1. **Train BigGAN-deep.** Its checkpoints used in our experiment are also provided in `eval_and_gan_ckpts`. Thus, to reproduce our results, the training of BigGAN is actually not necessary. <br />
BigGAN: Run `./ImageNet-100/GANs/BigGAN/scripts/launch_imagenet-100_deep.sh` <br />
2. **Implement each sampling method.**  Run `.sh` script(s) in the folder of each method. <br />
**cDR-RS and DRE-F-SP+RS**: Run `./scripts/run_exp_biggan.sh` for BigGAN. <br />
**DRS, DDLS, and Collab**: Run `./scripts/run_biggan.sh` for BigGAN. <br />




## 2. Sampling From CcGANs
### UTKFace (`./UTKFace`)
Download [eval_and_gan_ckpts.zip](https://1drv.ms/u/s!Arj2pETbYnWQuZpRaaDohsH5T0qFzg?e=jHYkIP). Unzip `eval_and_gan_ckpts.zip` to get `eval_and_gan_ckpts`, and move `eval_and_gan_ckpts` to `./UTKFace`.  This folder includes the checkpoint of AE and ResNet-34 for evaluation. It also includes the checkpoint of CcGAN (SVDL+ILI). <br />

Run `./scripts/run_train.sh` in each folder. <br />

### RC-49 (`./RC-49`)
Download [eval_and_gan_ckpts.zip](https://1drv.ms/u/s!Arj2pETbYnWQuZpShcQiZq8IzOFGQg?e=kyJlJI). Unzip `eval_and_gan_ckpts.zip` to get `eval_and_gan_ckpts`, and move `eval_and_gan_ckpts` to `./RC-49`. This folder includes the checkpoint of AE and ResNet-34 for evaluation. It also includes the checkpoint of CcGAN (SVDL+ILI). <br />

Run `./scripts/run_train.sh` in each folder. <br />


--------------------------------------------------------

# Computing NIQE of fake images sampled from CcGANs
Please refer to [https://github.com/UBCDingXin/improved_CcGAN](https://github.com/UBCDingXin/improved_CcGAN).

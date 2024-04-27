# RGB-to-NDVI Overview 

This repository uses a unet image training model to convert RGB images to NDVI images. The unet folder contains the training model. The RpiNDVI folder is python code to be used on a RasberryPi with a NoIR camera attached (sourced from https://github.com/robintw/RPiNDVI.git). In future versions two cameras will be used to simultaneously capture both the RGB and NDVI images to create a data set to train the model.

## Repository Structure

The repository is organized into two main directories, each dedicated to a specific model:

- [`UNet`](https://github.com/Menonlab-Rich/ml_models/tree/main/unet): Contains the implementation of the UNet model, designed for efficient image segmentation tasks. The directory includes Python scripts for the model architecture, dataset preparation, training routines, and utility functions. A detailed `readme.md` provides instructions and insights into the UNet model's implementation and usage.

- [`Pix2Pix`](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/tree/9f8f61e5a375c2e01c5187d093ce9c2409f409b0): Hosts the implementation of the Pix2Pix model, a popular framework for image-to-image translation tasks using conditional GANs. This section includes scripts for the generator and discriminator networks, configuration settings, dataset handling, and the training process.

## Getting Started
### UNet Directory

- [`config.py`](https://github.com/Menonlab-Rich/ml_models/blob/main/unet/config.py): Configuration settings for the UNet model training and evaluation.
- [`dataset.py`](https://github.com/Menonlab-Rich/ml_models/blob/main/unet/dataset.py): Dataset preparation and loading utilities for UNet.
- [`model.py`](https://github.com/Menonlab-Rich/ml_models/blob/main/unet/model.py): The UNet model architecture.
- [`requirements.txt`](https://github.com/Menonlab-Rich/ml_models/blob/main/unet/requirements.txt): Lists dependencies for running the UNet model.
- [`train.py`](https://github.com/Menonlab-Rich/ml_models/blob/main/unet/train.py): Script to train the UNet model.
- [`utils.py`](https://github.com/Menonlab-Rich/ml_models/blob/main/unet/utils.py): Utility functions supporting the UNet implementation.

### Pix2Pix Directory

## Prerequisites
- Linux or macOS
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN

## Getting Started
### Installation

- Clone this repo:
```bash
git clone https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
cd pytorch-CycleGAN-and-pix2pix
```

- Install [PyTorch](http://pytorch.org) and 0.4+ and other dependencies (e.g., torchvision, [visdom](https://github.com/facebookresearch/visdom) and [dominate](https://github.com/Knio/dominate)).
  - For pip users, please type the command `pip install -r requirements.txt`.
  - For Conda users, you can create a new Conda environment using `conda env create -f environment.yml`.
  - For Docker users, we provide the pre-built Docker image and Dockerfile. Please refer to our [Docker](docs/docker.md) page.
  - For Repl users, please click [![Run on Repl.it](https://repl.it/badge/github/junyanz/pytorch-CycleGAN-and-pix2pix)](https://repl.it/github/junyanz/pytorch-CycleGAN-and-pix2pix).

### CycleGAN train/test
- Download a CycleGAN dataset and change images in the file (e.g. maps):
```bash
bash ./datasets/download_cyclegan_dataset.sh maps
```
- To view training results and loss plots, run `python -m visdom.server` and click the URL http://localhost:8097.
- To log training progress and test images to W&B dashboard, set the `--use_wandb` flag with train and test script
- Train a model:
```bash
#!./scripts/train_cyclegan.sh
python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
```
To see more intermediate results, check out `./checkpoints/maps_cyclegan/web/index.html`.
- Test the model:
```bash
#!./scripts/test_cyclegan.sh
python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
```
- The test results will be saved to a html file here: `./results/maps_cyclegan/latest_test/index.html`.

### pix2pix train/test
- Download a pix2pix dataset (e.g.[facades](http://cmp.felk.cvut.cz/~tylecr1/facade/)):
```bash
bash ./datasets/download_pix2pix_dataset.sh facades
```
- To view training results and loss plots, run `python -m visdom.server` and click the URL http://localhost:8097.
- To log training progress and test images to W&B dashboard, set the `--use_wandb` flag with train and test script
- Train a model:
```bash
#!./scripts/train_pix2pix.sh
python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA
```
To see more intermediate results, check out  `./checkpoints/facades_pix2pix/web/index.html`.

- Test the model (`bash ./scripts/test_pix2pix.sh`):
```bash
#!./scripts/test_pix2pix.sh
python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA
```
- The test results will be saved to a html file here: `./results/facades_pix2pix/test_latest/index.html`. You can find more scripts at `scripts` directory.
- To train and test pix2pix-based colorization models, please add `--model colorization` and `--dataset_mode colorization`. See our training [tips](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md#notes-on-colorization) for more details.



## Citation
```
## The pix2pix model was created by:
The Link for code is:https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
@inproceedings{CycleGAN2017,
  title={Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks},
  author={Zhu, Jun-Yan and Park, Taesung and Isola, Phillip and Efros, Alexei A},
  booktitle={Computer Vision (ICCV), 2017 IEEE International Conference on},
  year={2017}
}


@inproceedings{isola2017image,
  title={Image-to-Image Translation with Conditional Adversarial Networks},
  author={Isola, Phillip and Zhu, Jun-Yan and Zhou, Tinghui and Efros, Alexei A},
  booktitle={Computer Vision and Pattern Recognition (CVPR), 2017 IEEE Conference on},
  year={2017}
}
```

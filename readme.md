# RGB-to-NDVI Overview 

This repository uses a unet image training model to convert RGB images to NDVI images. The unet folder contains the training model. The RpiNDVI folder is python code to be used on a RasberryPi with a NoIR camera attached (sourced from https://github.com/robintw/RPiNDVI.git). In future versions two cameras will be used to simultaneously capture both the RGB and NDVI images to create a data set to train the model.

## Repository Structure

The repository is organized into two main directories, each dedicated to a specific model:

- [`UNet`](https://github.com/Menonlab-Rich/ml_models/tree/main/unet): Contains the implementation of the UNet model, designed for efficient image segmentation tasks. The directory includes Python scripts for the model architecture, dataset preparation, training routines, and utility functions. A detailed `readme.md` provides instructions and insights into the UNet model's implementation and usage.

- [`RPiNDVI`](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/tree/9f8f61e5a375c2e01c5187d093ce9c2409f409b0): Hosts the implementation of the Pix2Pix model, a popular framework for image-to-image translation tasks using conditional GANs. This section includes scripts for the generator and discriminator networks, configuration settings, dataset handling, and the training process.
### UNet Directory

- [`config.py`](https://github.com/Menonlab-Rich/ml_models/blob/main/unet/config.py): Configuration settings for the UNet model training and evaluation.
- [`dataset.py`](https://github.com/Menonlab-Rich/ml_models/blob/main/unet/dataset.py): Dataset preparation and loading utilities for UNet.
- [`model.py`](https://github.com/Menonlab-Rich/ml_models/blob/main/unet/model.py): The UNet model architecture.
- [`requirements.txt`](https://github.com/Menonlab-Rich/ml_models/blob/main/unet/requirements.txt): Lists dependencies for running the UNet model.
- [`train.py`](https://github.com/Menonlab-Rich/ml_models/blob/main/unet/train.py): Script to train the UNet model.
- [`utils.py`](https://github.com/Menonlab-Rich/ml_models/blob/main/unet/utils.py): Utility functions supporting the UNet implementation.

### RPiNDVI Directory



## Getting Started


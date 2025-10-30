# 2D Improved UNet
## Project Overview

This project implements an Improved 2D U-Net architecture for multi-class semantic segmentation of prostate MRI slices from the HipMRI Study dataset.

The model was trained on over 11,000 MRI slices paired with their corresponding segmentation masks.
It replaces standard U-Net pooling layers with learnable stride-2 convolutions, adds Batch Normalization and LeakyReLU activations for smoother optimization, and incorporates Dropout regularization in the bottleneck.

## Model description





![alt text](image.png)
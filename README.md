# Vision-Transformer-and-Knowledge-Distillation-from-Scratch

This project implements the Vision Transformer architecture from *"An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"* and compares an optimization behavior with ResNet, in the condition of varying data size. Also conducts KD experiment, inspired by Data-efficient image Transformer, *"Training data-efficient image transformers & distillation through attention"* 

## What I implemented

### Vision Transformer Architecture
- Patch Embedding
- CLS Token
- Trainable Positional Encoding
- ViT Model


### Data-efficient image Transformer Architecture
- DIST Token
- DeiT Model
- Hard Label Distillation
- Late Fusion Inference Pipeline


### Experiment
- Training Procedure of ViT and ResNet adapted to 10K, 25K, 50K CIFAR10 datasets
- Accuracy comparison with ViT and DeiT distilled by ResNet

## Key Results

### Training with Varying Data Size: ViT vs ResNet

- ViT gains lower absolute train accuracy than ResNet, but benefits the increasing data size.

![vit_resnet_train_acc](./vit_resnet_train_acc.png)

### Knowledge Distillation: ViT vs DeiT

- DeiT gains marginally better performance(0.21%p) than ViT despite negligible extra parameters

| ViT | DeiT |
|---|---|
| 80.13% | 80.34% |


## Tech Stack

PyTorch, Torchvision, CUDA, Matplotlib, Jupyter Notebook

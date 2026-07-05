# Vision-Transformer-and-Knowledge-Distillation-from-Scratch

This project implements the Vision Transformer architecture from *"An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"* and compares its training behavior with ResNet under varying CIFAR-10 dataset sizes.

It also conducts a knowledge distillation experiment inspired by DeiT, *"Training Data-Efficient Image Transformers & Distillation Through Attention"*.

## What I implemented

### Vision Transformer

* Patch Embedding
* CLS Token
* Trainable Positional Embedding
* Vision Transformer Model

### Data-Efficient Image Transformer

* DIST Token
* DeiT-style Student Model
* Hard-label Distillation
* Late Fusion Inference Pipeline

### Experiments

* Training behavior comparison between ViT and ResNet on 10K, 25K, and 50K CIFAR-10 subsets
* Accuracy comparison between vanilla ViT and DeiT-style ViT distilled by ResNet

## Key Results

### Experiment 1: Training with Varying Dataset Size

ViT achieved lower absolute training accuracy than ResNet, but showed a larger gain as the dataset size increased.

This supports the observation that ViT has weaker visual inductive bias than CNNs and benefits more from increased data availability.

![vit_resnet_train_acc_gap](./results/vit_res_train_acc_gap.png)

### Experiment 2: Knowledge Distillation

DeiT achieved better performance than vanilla ViT by using a ResNet teacher and an additional distillation token.

| Model | Test Accuracy |
| ----- | ------------: |
| ViT   |        80.13% |
| DeiT  |        80.34% |

Accuracy gain:

```text
DeiT - ViT = +0.21%p
```

## Tech Stack

PyTorch, Torchvision, CUDA, Matplotlib, Jupyter Notebook

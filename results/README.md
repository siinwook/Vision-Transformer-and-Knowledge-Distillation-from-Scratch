# Experimental Results

This document summarizes the main experimental results and my observations.

Training conditions of experiments are indicated below

---

## 1. Adaption to Varying Data Size

| Train Accuracy | Train Loss |
| :---: | :---: |
| ![vit_res_train_acc.png](./vit_res_train_acc.png) | ![vit_res_train_loss.png](./vit_res_train_loss.png) |

| Increased Accuracy Gap| 
| :---: |
| ![vit_res_train_acc_gap.png](./vit_res_train_acc_gap.png) |

### Observation

- ResNet gains more absolute train accuracy and converges faster in any data sizes, ResNet trained with smallest data size wins ViT trained with biggest data size.
- As the data size increases, ViT shows much train accuracy growth rate(20.8%p+ vs 11.2%p+). 

### Interpretation

- Although the number of trainable parameters is similiar(ResNet: 1.14MB vs ViT: 1.19MB), ViT trained with biggest data size is inferior to ResNet with smallest data size due to the lack of inductive bias.
- As ViT gains more train accuracy growth rate when data size increases, ViT needs more data to learn inductive bias than CNNs.

---

## 2. Knowledge Distillation

| Model | Test Accuracy |
| ----- | ------------: |
| Teacher(ResNet)  |        91.46% |
| DeiT  |        80.34% |
| DeiT(only DIST token)  |        80.33% |
| DeiT(only CLS token)  |        80.24% |
| ViT   |        80.13% |


### Observation

- Models gain higher test accuracy in order of Teacher(ResNet), DeiT, and ViT.
- DeiT distilled by ConvNet teacher gains slightly more test accuracy. 

### Interpretation

- Inductive bias distillation from ConvNet to ViT helps ViT to get higher performance.
- Distillation token trained with ConvNet label gives more useful learning signal than classification token trained with ground truth.

---

**Training conditions**

| Conditions | Experiment 1 | Experiment 2 |
|---|---|---|
| Dataset | CIFAR-10 (10K, 25K, 50K) | CIFAR-10 (50K) |
| Model | ResNet74, ViT | ResNet74, ViT, DeiT |
| Optimizer | Adam | AdamW |
| Learning rate scheduler | Constant + Drop | Linear warmup + Cosine annealing |
| Learning rate | 0.001 (epoch 1-10), 0.0001 (epoch 11-50) | Linear warmup to 0.0005, then cosine annealing to 0.00001 |
| Batch size | 128 | 128 |
| Epochs | 50 | 40 |
| Data augmentation | Random crop, Horizontal flip, Color jitter, Gray scale, Normalize | Random crop, Horizontal flip, Color jitter, Gray scale, Normalize |
| Weight decay | 0 | 0.001 |
| Device | cuda | cuda |

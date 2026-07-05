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

- ResNet44 and PlainNet44 have almost similar parameter counts.
- ResNet44 has 0.661M trainable parameters
- PlainNet44 has 0.658M trainable parameters

### Interpretation

- ResNet44 requires marginally more parameters due to shortcut operations(Option B according to this paper), but the difference is negligible.

---

## 2. Knowledge Distillation

![resnet44 vs plainnet44 error_rate](./resnet44_vs_plainnet44_error_rate.png)

### Observation

- ResNet44 gains higher accuracy(by 9.0%) than PlainNet44
- ResNet44 starts convergence earlier than PlainNet44

### Interpretation

- Deep neural network with residual architecture is easier to optimize
- Skip connections used in every residual blocks, which represented as identity mapping, helps responses flow easier.  

---

**Training conditions**

| Conditions | Experiment 1 | Experiment 2 |
|---|---|---|
| Dataset | CIFAR-10 (10K, 25K, 50K) | CIFAR-10 (50K) |
| Optimizer | Adam | AdamW |
| Learning rate scheduler | Constant + Drop | Linear warmup + Cosine annealing |
| Learning rate | 0.001(1-10 epoch) + 0.0001(11-50 epoch) | 0.0005 to 0.00001(6-40 epoch) |
| Batch size | 128 | 128 |
| Epochs | 50 | 40 |
| Data augmentation | Random crop, Horizontal flip, Color jitter, Gray scale, Normalize | Random crop, Horizontal flip, Color jitter, Gray scale, Normalize |
| Weight decay | 0 | 0.001 |
| Device | cuda | cuda |

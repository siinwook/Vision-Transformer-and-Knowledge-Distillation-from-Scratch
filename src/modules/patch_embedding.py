import torch
from torch import nn
import math


class PatchEmbedding(nn.Module):
  def __init__(self, img_channel, patch_size, d_model):
    super().__init__()

    self.C = img_channel
    self.patch_size = patch_size

    self.Embedding = nn.Linear(patch_size**2 * img_channel, d_model)

  def forward(self, x):
    # (B,C,H,W)
    B,C,H,W = x.shape
    P = self.patch_size
    N = (H*W)//(P**2)

    out = x.new_zeros(B,N,P**2 * C) # same device with x
    idx=0
    for i in range(0, H, P):
      for j in range(0, W, P):
        out[:,idx,:] = x[:,:,i:i+P,j:j+P].reshape(B,-1)
        idx+=1
    out = self.Embedding(out) # (B,N,d_model)

    return out

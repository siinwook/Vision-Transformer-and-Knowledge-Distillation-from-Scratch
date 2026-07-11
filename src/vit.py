import torch
from torch import nn
import math

from .modules.patch_embedding import PatchEmbedding
from .modules.encoder import Encoder


class VisionTransformer(nn.Module):
  def __init__(self, pt_img_size, img_channel, pt_num_classes, ft_num_classes, patch_size, d_model, num_heads, d_ff, drop_out, L):
    super().__init__()

    self.PatchEmbedding = PatchEmbedding(img_channel, patch_size, d_model)

    self.x_cls = nn.Parameter(torch.randn(1,1,d_model) * math.sqrt(2.0 / d_model)) # He init

    pt_N = pt_img_size**2 // patch_size**2
    self.pt_PositionalEncoding = nn.Parameter(torch.randn(1,pt_N+1, d_model) * math.sqrt(2.0 / d_model)) # He init

    self.Encoder = Encoder(d_model, num_heads, d_ff, drop_out, L)

    self.pt_MLPHead = nn.Linear(d_model, pt_num_classes)
    self.ft_MLPHead = nn.Linear(d_model, ft_num_classes)


  def forward(self, x):
    B = x.size(0)

    x = self.PatchEmbedding(x) # (B,N,d_model)
    x = torch.cat((self.x_cls.expand(B,1,-1),x), dim=1) # (B,N+1,d_model)
    x = self.pt_PositionalEncoding + x

    x = self.Encoder(x)

    x = x[:,0,:]
    x = self.pt_MLPHead(x)

    return x

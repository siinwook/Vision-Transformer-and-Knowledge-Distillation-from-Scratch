import torch
from torch import nn
import math


class MultiheadSelfAttention(nn.Module):
  def __init__(self, d_model, num_heads):
    super().__init__()

    d_k = d_model // num_heads

    self.d_model = d_model
    self.d_k = d_k
    self.num_heads = num_heads

    self.W_Q = nn.Linear(d_model,d_model)
    self.W_K = nn.Linear(d_model,d_model)
    self.W_V = nn.Linear(d_model,d_model)
    self.W_O = nn.Linear(d_model,d_model)

  def forward(self,q,k,v,mask=None):
    batch_size = q.size(0)

    # q,k,v -> Q,K,V
    Q = self.W_Q(q)
    K = self.W_K(k)
    V = self.W_V(v)

    # Multi head
    Q = Q.reshape(batch_size,-1,self.num_heads,self.d_k).transpose(1,2)
    K = K.reshape(batch_size,-1,self.num_heads,self.d_k).transpose(1,2)
    V = V.reshape(batch_size,-1,self.num_heads,self.d_k).transpose(1,2)

    # Attention
    attn_score = torch.matmul(Q, K.transpose(-1,-2)) / math.sqrt(self.d_k)
    if mask is not None:
      attn_score = torch.masked_fill(attn_score, mask==0, -1e9) # mask = [1, 1, 1, ... 0, 0, 0]
    attn_score = torch.softmax(attn_score, dim=-1)
    out = torch.matmul(attn_score, V)

    # Concatenate
    out = out.transpose(1,2).reshape(batch_size,-1,self.d_model)
    out = self.W_O(out)

    return out

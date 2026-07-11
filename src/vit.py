import math

import torch
from torch import nn

from .modules import Encoder, PatchEmbedding


class VisionTransformer(nn.Module):
    """
    Vision Transformer for CIFAR-10 style image classification.

    Notebook-compatible constructor:
        VisionTransformer(
            pt_img_size=32,
            img_channel=3,
            pt_num_classes=10,
            ft_num_classes=10,
            patch_size=4,
            d_model=128,
            num_heads=4,
            d_ff=512,
            drop_out=0.1,
            L=6,
        )
    """

    def __init__(
        self,
        pt_img_size: int,
        img_channel: int,
        pt_num_classes: int,
        ft_num_classes: int,
        patch_size: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        drop_out: float,
        L: int,
    ):
        super().__init__()

        self.PatchEmbedding = PatchEmbedding(img_channel, patch_size, d_model)

        self.x_cls = nn.Parameter(
            torch.randn(1, 1, d_model) * math.sqrt(2.0 / d_model)
        )

        pt_num_patches = pt_img_size**2 // patch_size**2
        self.pt_PositionalEncoding = nn.Parameter(
            torch.randn(1, pt_num_patches + 1, d_model) * math.sqrt(2.0 / d_model)
        )

        self.Encoder = Encoder(d_model, num_heads, d_ff, drop_out, L)

        self.pt_MLPHead = nn.Linear(d_model, pt_num_classes)
        self.ft_MLPHead = nn.Linear(d_model, ft_num_classes)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)

        x = self.PatchEmbedding(x)
        x = torch.cat((self.x_cls.expand(batch_size, 1, -1), x), dim=1)
        x = self.pt_PositionalEncoding + x

        x = self.Encoder(x)
        cls_token = x[:, 0, :]

        return cls_token

    def forward(self, x: torch.Tensor, use_finetune_head: bool = False) -> torch.Tensor:
        cls_token = self.forward_features(x)

        if use_finetune_head:
            return self.ft_MLPHead(cls_token)

        return self.pt_MLPHead(cls_token)

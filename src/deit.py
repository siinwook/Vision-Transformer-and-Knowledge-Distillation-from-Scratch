import math
from typing import Tuple

import torch
from torch import nn

from .modules import Encoder, PatchEmbedding


class DeiT(nn.Module):
    """
    DeiT-style Vision Transformer with a distillation token.

    Forward output:
        (cls_logits, dist_logits)

    Inference can use late fusion:
        softmax(cls_logits) + softmax(dist_logits)
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
        self.x_dist = nn.Parameter(
            torch.randn(1, 1, d_model) * math.sqrt(2.0 / d_model)
        )

        pt_num_patches = pt_img_size**2 // patch_size**2
        self.pt_PositionalEncoding = nn.Parameter(
            torch.randn(1, pt_num_patches + 2, d_model) * math.sqrt(2.0 / d_model)
        )

        self.Encoder = Encoder(d_model, num_heads, d_ff, drop_out, L)

        self.pt_cls_MLPHead = nn.Linear(d_model, pt_num_classes)
        self.pt_dist_MLPHead = nn.Linear(d_model, pt_num_classes)
        self.ft_MLPHead = nn.Linear(d_model, ft_num_classes)

    def forward_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.size(0)

        x = self.PatchEmbedding(x)
        x = torch.cat(
            (
                self.x_cls.expand(batch_size, 1, -1),
                self.x_dist.expand(batch_size, 1, -1),
                x,
            ),
            dim=1,
        )
        x = self.pt_PositionalEncoding + x

        x = self.Encoder(x)

        cls_token = x[:, 0, :]
        dist_token = x[:, 1, :]

        return cls_token, dist_token

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        cls_token, dist_token = self.forward_features(x)

        cls_logits = self.pt_cls_MLPHead(cls_token)
        dist_logits = self.pt_dist_MLPHead(dist_token)

        return cls_logits, dist_logits

    @torch.no_grad()
    def predict_late_fusion(self, x: torch.Tensor) -> torch.Tensor:
        cls_logits, dist_logits = self.forward(x)
        fused_logits = torch.softmax(cls_logits, dim=-1) + torch.softmax(
            dist_logits, dim=-1
        )

        return fused_logits.argmax(dim=-1)

from typing import Dict, Optional

import torch
from torch import nn


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def evaluate_classifier(
    model: nn.Module,
    test_loader,
    device: Optional[torch.device] = None,
) -> float:
    """
    Evaluate ResNet or vanilla ViT on a classification dataset.
    """

    if device is None:
        device = get_device()

    model = model.to(device)
    model.eval()

    correct = 0
    total = 0

    for x_test, y_test in test_loader:
        x_test = x_test.to(device)
        y_test = y_test.to(device)

        logits = model(x_test)
        if isinstance(logits, (tuple, list)):
            logits = logits[0]

        correct += (logits.argmax(dim=-1) == y_test).sum().item()
        total += y_test.size(0)

    return correct / total


@torch.no_grad()
def evaluate_deit(
    model: nn.Module,
    test_loader,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """
    Evaluate DeiT using:
    - late fusion of CLS and DIST predictions
    - CLS token only
    - DIST token only
    """

    if device is None:
        device = get_device()

    model = model.to(device)
    model.eval()

    fused_correct = 0
    cls_correct = 0
    dist_correct = 0
    total = 0

    for x_test, y_test in test_loader:
        x_test = x_test.to(device)
        y_test = y_test.to(device)

        cls_logits, dist_logits = model(x_test)

        fused_logits = torch.softmax(cls_logits, dim=-1) + torch.softmax(
            dist_logits, dim=-1
        )

        fused_correct += (fused_logits.argmax(dim=-1) == y_test).sum().item()
        cls_correct += (cls_logits.argmax(dim=-1) == y_test).sum().item()
        dist_correct += (dist_logits.argmax(dim=-1) == y_test).sum().item()
        total += y_test.size(0)

    return {
        "deit_acc": fused_correct / total,
        "cls_acc": cls_correct / total,
        "dist_acc": dist_correct / total,
    }


@torch.no_grad()
def predict_deit_late_fusion(
    model: nn.Module,
    x: torch.Tensor,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Return DeiT late-fusion class predictions for a batch of images.
    """

    if device is None:
        device = get_device()

    model = model.to(device)
    model.eval()

    x = x.to(device)
    cls_logits, dist_logits = model(x)

    fused_logits = torch.softmax(cls_logits, dim=-1) + torch.softmax(
        dist_logits, dim=-1
    )

    return fused_logits.argmax(dim=-1)

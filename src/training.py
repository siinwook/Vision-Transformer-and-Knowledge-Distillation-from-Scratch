from typing import Dict, Optional

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def classification_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return (logits.argmax(dim=-1) == labels).float().mean()


def build_warmup_cosine_scheduler(
    optimizer: Optimizer,
    warmup_epochs: int = 5,
    total_epochs: int = 40,
    eta_min: float = 1e-5,
) -> LRScheduler:
    """
    Linear warmup followed by cosine annealing.

    Matches the Experiment 2 setup:
    - LinearLR for 5 epochs
    - CosineAnnealingLR for 35 epochs when total_epochs=40
    """

    cosine_epochs = total_epochs - warmup_epochs

    scheduler_linear = torch.optim.lr_scheduler.LinearLR(
        optimizer=optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=warmup_epochs,
    )

    scheduler_cos = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer,
        T_max=cosine_epochs,
        eta_min=eta_min,
    )

    return torch.optim.lr_scheduler.SequentialLR(
        optimizer=optimizer,
        schedulers=[scheduler_linear, scheduler_cos],
        milestones=[warmup_epochs],
    )


def train_classifier(
    model: nn.Module,
    train_loader,
    optimizer: Optimizer,
    loss_fn: Optional[nn.Module] = None,
    scheduler: Optional[LRScheduler] = None,
    epochs: int = 1,
    device: Optional[torch.device] = None,
) -> Dict[str, list]:
    """
    Train a standard classifier such as ResNet or ViT.

    Returns:
        history = {"loss": [...], "acc": [...]}
    """

    if device is None:
        device = get_device()

    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss()

    model = model.to(device)
    history = {"loss": [], "acc": []}

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0

        for x_train, y_train in train_loader:
            x_train = x_train.to(device)
            y_train = y_train.to(device)

            logits = model(x_train)
            if isinstance(logits, (tuple, list)):
                logits = logits[0]

            optimizer.zero_grad()
            loss = loss_fn(logits, y_train)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            correct += (logits.argmax(dim=-1) == y_train).sum().item()
            total += y_train.size(0)

        if scheduler is not None:
            scheduler.step()

        avg_loss = epoch_loss / len(train_loader)
        avg_acc = correct / total

        history["loss"].append(avg_loss)
        history["acc"].append(avg_acc)

        print(f"epoch {epoch + 1} loss: {avg_loss:.6f} acc: {avg_acc:.6f}")

    return history


def train_deit_hard_distillation(
    student: nn.Module,
    teacher: nn.Module,
    train_loader,
    optimizer: Optimizer,
    loss_fn: Optional[nn.Module] = None,
    scheduler: Optional[LRScheduler] = None,
    epochs: int = 1,
    alpha: float = 0.5,
    device: Optional[torch.device] = None,
) -> Dict[str, list]:
    """
    Train DeiT with hard-label distillation.

    Loss:
        (1 - alpha) * CE(cls_logits, ground_truth)
        + alpha * CE(dist_logits, teacher_argmax)
    """

    if device is None:
        device = get_device()

    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss()

    student = student.to(device)
    teacher = teacher.to(device)
    teacher.eval()

    history = {"loss": [], "cls_acc": [], "dist_acc": []}

    for epoch in range(epochs):
        student.train()
        epoch_loss = 0.0
        cls_correct = 0
        dist_correct = 0
        total = 0

        for x_train, y_train in train_loader:
            x_train = x_train.to(device)
            y_train = y_train.to(device)

            cls_logits, dist_logits = student(x_train)

            with torch.no_grad():
                teacher_labels = teacher(x_train).argmax(dim=-1)

            loss = (1.0 - alpha) * loss_fn(cls_logits, y_train)
            loss = loss + alpha * loss_fn(dist_logits, teacher_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            cls_correct += (cls_logits.argmax(dim=-1) == y_train).sum().item()
            dist_correct += (dist_logits.argmax(dim=-1) == y_train).sum().item()
            total += y_train.size(0)

        if scheduler is not None:
            scheduler.step()

        avg_loss = epoch_loss / len(train_loader)
        cls_acc = cls_correct / total
        dist_acc = dist_correct / total

        history["loss"].append(avg_loss)
        history["cls_acc"].append(cls_acc)
        history["dist_acc"].append(dist_acc)

        print(
            f"epoch {epoch + 1} loss: {avg_loss:.6f} "
            f"cls_acc: {cls_acc:.6f} dist_acc: {dist_acc:.6f}"
        )

    return history

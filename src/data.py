from typing import Dict, Iterable, Optional

import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10
from torchvision.transforms import v2


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def build_cifar10_transforms(normalize: bool = False):
    """
    Build CIFAR-10 train/test transforms used in the notebooks.

    normalize=False matches the uploaded notebook.
    Set normalize=True if you want standard CIFAR-10 normalization.
    """

    train_transforms = [
        v2.RandomCrop(32, padding=4),
        v2.RandomHorizontalFlip(),
        v2.ColorJitter(0.4, 0.4, 0.4, 0.1),
        v2.RandomGrayscale(0.2),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ]

    test_transforms = [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ]

    if normalize:
        train_transforms.append(v2.Normalize(CIFAR10_MEAN, CIFAR10_STD))
        test_transforms.append(v2.Normalize(CIFAR10_MEAN, CIFAR10_STD))

    return v2.Compose(train_transforms), v2.Compose(test_transforms)


def build_cifar10_datasets(
    root: str = "./data",
    download: bool = True,
    normalize: bool = False,
    use_mirror: bool = False,
):
    """
    Build CIFAR-10 train/test datasets.

    use_mirror=True uses the mirror URL from the notebook.
    """

    if use_mirror:
        CIFAR10.url = (
            "https://data.brainchip.com/dataset-mirror/cifar10/"
            "cifar-10-python.tar.gz"
        )

    train_transform, test_transform = build_cifar10_transforms(normalize=normalize)

    train_dataset = CIFAR10(
        root=root,
        train=True,
        transform=train_transform,
        download=download,
    )

    test_dataset = CIFAR10(
        root=root,
        train=False,
        transform=test_transform,
        download=download,
    )

    return train_dataset, test_dataset


def build_cifar10_dataloaders(
    root: str = "./data",
    batch_size: int = 128,
    num_workers: int = 2,
    dataset_sizes: Iterable[int] = (10000, 25000, 50000),
    download: bool = True,
    normalize: bool = False,
    use_mirror: bool = False,
) -> Dict[str, DataLoader]:
    """
    Build CIFAR-10 dataloaders for 10K / 25K / 50K experiments.

    Returns:
        {
            "train_10K": ...,
            "train_25K": ...,
            "train_50K": ...,
            "test": ...
        }
    """

    train_dataset, test_dataset = build_cifar10_datasets(
        root=root,
        download=download,
        normalize=normalize,
        use_mirror=use_mirror,
    )

    dataloaders = {}

    for size in dataset_sizes:
        subset = Subset(train_dataset, list(range(size)))
        key = f"train_{size // 1000}K"
        dataloaders[key] = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    dataloaders["test"] = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return dataloaders

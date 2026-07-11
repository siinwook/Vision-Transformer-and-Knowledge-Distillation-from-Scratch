import torch
from torch import nn


class ResNet(nn.Module):
    """
    CIFAR-10 ResNet teacher model used in the KD experiment.

    This implementation follows the notebook structure:
    - Initial 3x3 Conv
    - Three residual stages
    - Option B projection when feature map size changes
    - Global average pooling
    - Linear classifier

    For N=12, this corresponds to ResNet74 in the notebook.
    """

    def __init__(self, N: int, num_classes: int = 10):
        super().__init__()

        self.N = N
        filter_size = 16

        self.Pre = nn.Sequential(
            nn.Conv2d(3, 16, (3, 3), padding=1, bias=False),
            nn.BatchNorm2d(filter_size),
            nn.ReLU(),
        )

        module_dict = {}

        for i in range(1, 4):
            for n in range(1, self.N * 2 + 1):
                if i != 1 and n == 1:
                    module_dict[f"Change_Dim{i}"] = nn.Sequential(
                        nn.Conv2d(
                            filter_size,
                            filter_size * 2,
                            (1, 1),
                            stride=2,
                            bias=False,
                        ),
                        nn.BatchNorm2d(filter_size * 2),
                    )
                    module_dict[f"Conv{i}_{n}"] = nn.Conv2d(
                        filter_size,
                        filter_size * 2,
                        (3, 3),
                        stride=2,
                        padding=1,
                        bias=False,
                    )
                    filter_size *= 2
                else:
                    module_dict[f"Conv{i}_{n}"] = nn.Conv2d(
                        filter_size,
                        filter_size,
                        (3, 3),
                        stride=1,
                        padding=1,
                        bias=False,
                    )

                module_dict[f"Batchnorm{i}_{n}"] = nn.BatchNorm2d(filter_size)
                module_dict[f"ReLU{i}_{n}"] = nn.ReLU()

        module_dict["Avgpool"] = nn.AdaptiveAvgPool2d((1, 1))
        module_dict["Flatten"] = nn.Flatten()
        module_dict["FC"] = nn.Linear(64, num_classes)

        self.ModuleDict = nn.ModuleDict(module_dict)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.Pre(x)

        for i in range(1, 4):
            for n in range(1, self.N * 2 + 1, 2):
                y = self.ModuleDict[f"Conv{i}_{n}"](x)
                y = self.ModuleDict[f"Batchnorm{i}_{n}"](y)
                y = self.ModuleDict[f"ReLU{i}_{n}"](y)

                y = self.ModuleDict[f"Conv{i}_{n + 1}"](y)
                y = self.ModuleDict[f"Batchnorm{i}_{n + 1}"](y)

                if i != 1 and n == 1:
                    x = self.ModuleDict[f"Change_Dim{i}"](x)

                x = self.ModuleDict[f"ReLU{i}_{n + 1}"](y + x)

        x = self.ModuleDict["Avgpool"](x)
        x = self.ModuleDict["Flatten"](x)
        x = self.ModuleDict["FC"](x)

        return x

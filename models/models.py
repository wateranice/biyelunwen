import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """
    轻量 CNN：通过 ``in_channels`` / ``input_size`` 适配 CIFAR-10(32×32×3) 与 MNIST/Fashion-MNIST(28×28×1)。
    三次 MaxPool(2) 后空间边长为 ``input_size // 8``。
    """

    def __init__(self, in_channels: int = 3, num_classes: int = 10, input_size: int = 32):
        super().__init__()
        self.in_channels = in_channels
        self.input_size = input_size
        dim = input_size // 8
        flat_dim = 64 * dim * dim

        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(flat_dim, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

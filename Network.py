import torch.nn as nn
import torch.nn.functional as F


class Network(nn.Module):

    def __init__(self):
        super(Network, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=3, padding=1)     # in (161,101) out (161,101)
        self.norm1 = nn.BatchNorm2d(6)

        # here - maxPool

        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=3, padding=1)   # in (80,50) out (80,50)
        self.norm2 = nn.BatchNorm2d(12)

        # here - maxPool

        self.conv3 = nn.Conv2d(in_channels=12, out_channels=16, kernel_size=3, padding=1)  # in (40,25) out (40,25)
        self.norm3 = nn.BatchNorm2d(16)

        # here - maxPool

        self.fc1 = nn.Linear(in_features=16 * 8 * 5, out_features=120)
        self.norm4 = nn.BatchNorm1d(120)

        self.fc2 = nn.Linear(in_features=120, out_features=60)
        self.norm5 = nn.BatchNorm1d(60)

        self.out = nn.Linear(in_features=60, out_features=30)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.norm1(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.norm2(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.norm3(x)
        x = F.max_pool2d(x, kernel_size=5, stride=5)

        x = x.view(-1, 16 * 8 * 5)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.norm4(x)

        x = self.fc2(x)
        x = F.relu(x)
        x = self.norm5(x)

        x = self.out(x)

        return x

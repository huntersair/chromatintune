import torch
import torch.nn as nn


class SequenceCNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.conv1 = nn.Conv1d(
            in_channels=4,
            out_channels=32,
            kernel_size=5
        )

        self.relu = nn.ReLU()

        self.pool = nn.AdaptiveMaxPool1d(1)

        self.fc = nn.Linear(32, 1)

    def forward(self, x):

        x = self.conv1(x)

        x = self.relu(x)

        x = self.pool(x)

        x = x.squeeze(-1)

        x = self.fc(x)

        return x
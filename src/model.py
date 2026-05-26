import torch
import torch.nn as nn


class SequenceCNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.conv1 = nn.Conv1d(
            in_channels=4,
            out_channels=32,
            kernel_size=11
        )

        self.relu = nn.ReLU()

        self.pool = nn.AdaptiveMaxPool1d(1)

        # 32 CNN features
        # + 1 H3K27ac feature
        self.fc = nn.Linear(33, 1)

    def forward(
        self,
        sequence,
        h3k27ac
    ):

        x = self.conv1(sequence)

        x = self.relu(x)

        x = self.pool(x)

        x = x.squeeze(-1)

        # concatenate epigenomic feature
        x = torch.cat(
            [x, h3k27ac],
            dim=1
        )

        x = self.fc(x)

        return x
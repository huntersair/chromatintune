import torch
import torch.nn as nn


class SequenceCNN(nn.Module):

    def __init__(
        self,
        use_h3k27ac=True
    ):

        super().__init__()

        self.use_h3k27ac = use_h3k27ac

        self.conv1 = nn.Conv1d(
            in_channels=4,
            out_channels=64,
            kernel_size=11
        )

        self.relu = nn.ReLU()

        self.pool = nn.AdaptiveMaxPool1d(1)

        if self.use_h3k27ac:

            self.fc = nn.Linear(
                65,
                1
            )

        else:

            self.fc = nn.Linear(
                64,
                1
            )

    def forward(
        self,
        sequence,
        h3k27ac=None
    ):

        x = self.conv1(sequence)

        x = self.relu(x)

        x = self.pool(x)

        x = x.squeeze(-1)

        if self.use_h3k27ac:

            x = torch.cat(
                [x, h3k27ac],
                dim=1
            )

        x = self.fc(x)

        return x

class DeepSequenceCNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.conv1 = nn.Conv1d(
            in_channels=4,
            out_channels=64,
            kernel_size=11,
            padding=5
        )

        self.conv2 = nn.Conv1d(
            in_channels=64,
            out_channels=128,
            kernel_size=7,
            padding=3
        )

        self.relu = nn.ReLU()

        self.pool = nn.AdaptiveMaxPool1d(1)

        self.dropout = nn.Dropout(
            p=0.3
        )

        self.fc1 = nn.Linear(
            128,
            64
        )

        self.fc2 = nn.Linear(
            64,
            1
        )

    def forward(
        self,
        sequence
    ):

        x = self.conv1(sequence)

        x = self.relu(x)

        x = self.conv2(x)

        x = self.relu(x)

        x = self.pool(x)

        x = x.squeeze(-1)

        x = self.dropout(x)

        x = self.fc1(x)

        x = self.relu(x)

        x = self.fc2(x)

        return x
import torch.nn as nn


class H3K27acModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(1, 16),

            nn.ReLU(),

            nn.Linear(16, 1)
        )

    def forward(self, h3k27ac):

        return self.network(h3k27ac)
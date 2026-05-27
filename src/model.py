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

class SequenceTransformer(nn.Module):

    def __init__(self):

        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=4,
            embedding_dim=64
        )

        self.positional_embedding = nn.Embedding(
            num_embeddings=200,
            embedding_dim=64
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=64,
            nhead=4,
            dropout=0.1,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2
        )

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Linear(
            64,
            1
        )

    def forward(
        self,
        tokens
    ):

        positions = torch.arange(
            0,
            tokens.size(1),
            device=tokens.device
        ).unsqueeze(0)

        x = self.embedding(tokens)

        x = x + self.positional_embedding(positions)

        x = self.transformer(x)

        x = x.transpose(1, 2)

        x = self.pool(x)

        x = x.squeeze(-1)

        x = self.fc(x)

        return x
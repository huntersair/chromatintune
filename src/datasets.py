import pandas as pd
import torch
import numpy as np

from torch.utils.data import Dataset

from src.utils import one_hot_encode


class SequenceDataset(Dataset):

    def __init__(
        self,
        csv_path: str,
        max_length: int = 200
    ):

        self.dataframe = pd.read_csv(csv_path)

        self.max_length = max_length

    def __len__(self):

        return len(self.dataframe)

    def __getitem__(self, idx):

        row = self.dataframe.iloc[idx]

        sequence = row["sequence"]
        expression = row["expression"]

        encoded_sequence = one_hot_encode(sequence)

        encoded_sequence = self.pad_sequence(
            encoded_sequence
        )

        sequence_tensor = torch.tensor(
            encoded_sequence,
            dtype=torch.float32
        ).permute(1, 0)

        expression_tensor = torch.tensor(
            [expression],
            dtype=torch.float32
        )

        return sequence_tensor, expression_tensor

    def pad_sequence(self, encoded_sequence):

        sequence_length = encoded_sequence.shape[0]

        if sequence_length > self.max_length:

            encoded_sequence = encoded_sequence[
                :self.max_length
            ]

        elif sequence_length < self.max_length:

            padding_length = (
                    self.max_length - sequence_length
            )

            padding = np.zeros(
                (padding_length, 4),
                dtype=np.float32
            )

            encoded_sequence = np.vstack(
                [encoded_sequence, padding]
            )

        return encoded_sequence
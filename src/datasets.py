import pandas as pd
import torch

from torch.utils.data import Dataset

from src.utils import one_hot_encode


class SequenceDataset(Dataset):

    def __init__(self, csv_path: str):

        self.dataframe = pd.read_csv(csv_path)

    def __len__(self):

        return len(self.dataframe)

    def __getitem__(self, idx):

        row = self.dataframe.iloc[idx]

        sequence = row["sequence"]
        expression = row["expression"]

        encoded_sequence = one_hot_encode(sequence)

        sequence_tensor = torch.tensor(
            encoded_sequence,
            dtype=torch.float32
        ).permute(1, 0)

        expression_tensor = torch.tensor(
            [expression],
            dtype=torch.float32
        )

        return sequence_tensor, expression_tensor
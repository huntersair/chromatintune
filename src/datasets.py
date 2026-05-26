import pandas as pd
import torch

from torch.utils.data import Dataset

from src.utils import one_hot_encode


class SequenceDataset(Dataset):

    def __init__(
        self,
        csv_path: str
    ):

        self.dataframe = pd.read_csv(csv_path)

    def __len__(self):

        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]

        sequence = row["sequence"]

        label = row["label"]

        chrom = row["chrom"]

        h3k27ac = row["h3k27ac"]

        encoded_sequence = one_hot_encode(
            sequence
        )

        encoded_sequence = torch.tensor(
            encoded_sequence,
            dtype=torch.float32
        )

        encoded_sequence = encoded_sequence.permute(1, 0)

        label = torch.tensor(
            [label],
            dtype=torch.float32
        )

        h3k27ac = torch.tensor(
            [h3k27ac],
            dtype=torch.float32
        )

        return (
            encoded_sequence,
            h3k27ac,
            label,
            chrom
        )
import torch

from torch.utils.data import Dataset

from src.utils import (
    tokenize_sequence,
    reverse_complement
)


class TransformerMPRADataset(Dataset):

    def __init__(
        self,
        dataframe,
        augment=False
    ):

        self.dataframe = dataframe

        self.augment = augment

    def __len__(self):

        return len(self.dataframe)

    def __getitem__(self, idx):

        row = self.dataframe.iloc[idx]

        sequence = row["SEQUENCE"]

        activity = row["HepG2_lfc"]

        if self.augment:

            if torch.rand(1).item() < 0.5:

                sequence = reverse_complement(
                    sequence
                )

        tokens = tokenize_sequence(
            sequence
        )

        tokens = torch.tensor(
            tokens,
            dtype=torch.long
        )

        activity = torch.tensor(
            activity,
            dtype=torch.float32
        )

        return tokens, activity
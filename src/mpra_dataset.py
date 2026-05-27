import torch
from torch.utils.data import Dataset
from src.utils import reverse_complement
from src.utils import one_hot_encode


class MPRADataset(Dataset):

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

        if self.augment:

            if torch.rand(1).item() < 0.5:
                sequence = reverse_complement(
                    sequence
                )

        activity = row["HepG2_lfc"]

        encoded = one_hot_encode(
            sequence
        )

        encoded = torch.tensor(
            encoded,
            dtype=torch.float32
        )

        encoded = encoded.permute(1, 0)

        activity = torch.tensor(
            activity,
            dtype=torch.float32
        )

        return encoded, activity
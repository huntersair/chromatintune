import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import mlflow
import torch
import numpy as np
import matplotlib.pyplot as plt

from src.datasets import SequenceDataset
from src.model import SequenceCNN


dataset = SequenceDataset(
    "data/processed/liver_accessibility_multimodal.csv"
)

dataset_df = dataset.dataframe

model = SequenceCNN()

model.load_state_dict(
    torch.load(
        "models/liver_accessibility_cnn.pth"
    )
)

model.eval()

KERNEL_SIZE = 11

nucleotide_to_index = {
    "A": 0,
    "C": 1,
    "G": 2,
    "T": 3
}

os.makedirs(
    "figures",
    exist_ok=True
)

mlflow.set_experiment(
    "filter_interpretability"
)

with mlflow.start_run():

    mlflow.log_param(
        "kernel_size",
        KERNEL_SIZE
    )

    mlflow.log_param(
        "num_filters",
        32
    )

    for FILTER_INDEX in range(32):

        top_motifs = []

        for i in range(1000):

            sample_x, _, sample_y, chrom = dataset[i]

            sample_x = sample_x.unsqueeze(0)

            conv_output = model.conv1(
                sample_x
            )

            activations = (
                conv_output[
                    0,
                    FILTER_INDEX
                ]
                .detach()
                .numpy()
            )

            max_position = activations.argmax()

            max_activation = activations.max()

            sequence = dataset_df.iloc[i][
                "sequence"
            ]

            motif_candidate = sequence[
                max_position:max_position + KERNEL_SIZE
            ]

            top_motifs.append({
                "motif": motif_candidate,
                "activation": max_activation
            })

        top_motifs = sorted(
            top_motifs,
            key=lambda x: x["activation"],
            reverse=True
        )

        seen = set()

        unique_top_motifs = []

        for entry in top_motifs:

            motif = entry["motif"]

            if motif not in seen:

                unique_top_motifs.append(
                    entry
                )

                seen.add(motif)

        top_sequences = [
            entry["motif"]
            for entry in unique_top_motifs[:100]
        ]

        pfm = np.zeros(
            (4, KERNEL_SIZE)
        )

        for sequence in top_sequences:

            for position, nucleotide in enumerate(sequence):

                row = nucleotide_to_index[
                    nucleotide
                ]

                pfm[row, position] += 1

        pfm = pfm / pfm.sum(axis=0)

        plt.figure(figsize=(14, 3))

        plt.imshow(pfm)

        plt.yticks(
            [0, 1, 2, 3],
            ["A", "C", "G", "T"]
        )

        plt.xticks(
            range(KERNEL_SIZE),
            range(1, KERNEL_SIZE + 1)
        )

        plt.xlabel(
            "Motif Position"
        )

        plt.ylabel(
            "Nucleotide"
        )

        plt.colorbar(
            label="Frequency"
        )

        plt.title(
            f"Filter {FILTER_INDEX} Position Frequency Matrix"
        )

        save_path = (
            f"figures/filter_{FILTER_INDEX}.png"
        )

        plt.savefig(save_path)

        mlflow.log_artifact(
            save_path,
            artifact_path="motifs"
        )

        plt.close()

        print(
            f"Completed filter {FILTER_INDEX}"
        )
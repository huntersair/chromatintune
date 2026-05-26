import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

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

SAMPLE_INDEX = 0

sample_x, sample_h3k27ac, sample_y, chrom = dataset[
    SAMPLE_INDEX
]

sequence = dataset_df.iloc[
    SAMPLE_INDEX
]["sequence"]

sample_x = sample_x.unsqueeze(0)

sample_h3k27ac = sample_h3k27ac.unsqueeze(0)

sample_x.requires_grad = True

prediction = model(
    sample_x,
    sample_h3k27ac
)

prediction.backward()

gradients = (
    sample_x.grad
    .detach()
    .numpy()[0]
)

importance_scores = np.abs(
    gradients
).max(axis=0)

plt.figure(figsize=(18, 4))

plt.plot(importance_scores)

plt.xlabel("Sequence Position")

plt.ylabel("Importance")

plt.title(
    "Saliency Map"
)

top_positions = np.argsort(
    importance_scores
)[-20:]

print("\nTop Important Positions:\n")

for pos in top_positions:

    print(
        f"Position {pos} | "
        f"Base: {sequence[pos]} | "
        f"Importance: "
        f"{importance_scores[pos]:.4f}"
    )


WINDOW_SIZE = 11

print("\nImportant Motif Windows:\n")

for pos in top_positions:

    start = max(
        0,
        pos - WINDOW_SIZE // 2
    )

    end = min(
        len(sequence),
        pos + WINDOW_SIZE // 2
    )

    motif_window = sequence[start:end]

    print(
        f"Position {pos} | "
        f"Window: {motif_window}"
    )


plt.savefig(
    "./figures/saliency_map.png"
)

plt.close()
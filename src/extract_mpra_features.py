import torch
import pandas as pd
from src.model import RegulatoryResNet
from src.utils import one_hot_encode
import numpy as np
model = RegulatoryResNet(dropout=0.0,se_reduction=16)

model.load_state_dict(
    torch.load(
        "../models/bayesopt_regulatory_resnet.pth"
    )
)

merged_df = pd.read_csv("../data/processed/mpra_hepg2_processed.csv")

model.eval()

atac_preds = []
h3k27ac_preds = []
embeddings = []

with torch.no_grad():

    for sequence in merged_df["SEQUENCE"]:

        encoded_seq = one_hot_encode(
            sequence
        )

        encoded_seq = encoded_seq.unsqueeze(0)

        atac_pred, h3k27ac_pred, embedding = model(
            encoded_seq
        )

        atac_preds.append(
            atac_pred.item()
        )

        h3k27ac_preds.append(
            h3k27ac_pred.item()
        )

        embeddings.append(
            embedding.squeeze(0).numpy()
        )

    embeddings = np.vstack(
        embeddings
    )

    print(
        embeddings.shape
    )

    merged_df["pred_atac"] = atac_preds

    merged_df["pred_h3k27ac"] = h3k27ac_preds

    merged_df.to_csv(
        "mpra_features.csv",
        index=False
    )

    np.save(
        "mpra_embeddings.npy",
        embeddings
    )
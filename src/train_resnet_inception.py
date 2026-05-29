import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import mlflow

import numpy as np
import random

import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.utils.data import random_split

import pandas as pd

from scipy.stats import pearsonr

from sklearn.metrics import roc_auc_score

from tqdm import tqdm

from src.model import RegulatoryResNet
from src.datasets import MultimodalDataset


def train_model(
    lr,
    weight_decay,
    dropout,
    se_reduction,
    lambda_h3k27ac,
    train_fraction=0.3,
    epochs=5,
    save_model=True
):

    mlflow.set_experiment(
        "RegulatoryResNet"
    )

    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    with mlflow.start_run():

        df = pd.read_csv(
            "./data/processed/liver_accessibility_multimodal_rc.csv"
        )

        print("Dataset loaded")

        dataset = MultimodalDataset(df)

        if train_fraction < 1.0:
            subset_size = int(
                len(dataset) * train_fraction
            )

            dataset, _ = random_split(
                dataset,
                [
                    subset_size,
                    len(dataset) - subset_size
                ]
            )

        train_size = int(
            0.8 * len(dataset)
        )

        val_size = len(dataset) - train_size

        generator = torch.Generator().manual_seed(42)

        train_dataset, val_dataset = random_split(
            dataset,
            [train_size, val_size],
            generator=generator
        )

        train_loader = DataLoader(

            train_dataset,

            batch_size=32,

            shuffle=True,

            num_workers=0

        )

        val_loader = DataLoader(

            val_dataset,

            batch_size=32,

            num_workers=0

        )

        print("Dataloaders created")

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)

        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        print(f"Device: {device}")

        model = RegulatoryResNet(
            dropout=dropout,
            se_reduction=se_reduction
        ).to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=3
        )

        atac_criterion = nn.BCEWithLogitsLoss()

        h3k27ac_criterion = nn.MSELoss()

        best_auc = 0

        best_pearson = -1

        best_score = -float("inf")

        patience = 5

        patience_counter = 0

        print("Train start.")

        for epoch in range(epochs):

            model.train()

            train_loss = 0

            train_bar = tqdm(
                train_loader,
                desc=f"Epoch {epoch+1}"
            )

            for (
                sequences,
                atac_labels,
                h3k27ac_targets
            ) in train_bar:

                sequences = sequences.to(
                    device
                )

                atac_labels = atac_labels.to(
                    device
                )

                h3k27ac_targets = h3k27ac_targets.to(
                    device
                )

                optimizer.zero_grad()

                (
                    atac_logits,
                    h3k27ac_output,
                    embeddings
                ) = model(sequences)

                atac_logits = atac_logits.squeeze(-1)

                h3k27ac_output = h3k27ac_output.squeeze(-1)

                atac_loss = atac_criterion(
                    atac_logits,
                    atac_labels
                )

                h3k27ac_loss = h3k27ac_criterion(
                    h3k27ac_output,
                    h3k27ac_targets
                )

                total_loss = (
                        atac_loss
                        + lambda_h3k27ac * h3k27ac_loss
                )

                total_loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    1.0
                )

                optimizer.step()

                train_loss += total_loss.item()

            model.eval()

            val_loss = 0

            all_atac_probs = []

            all_atac_labels = []

            all_h3k27ac_preds = []

            all_h3k27ac_targets = []

            with torch.no_grad():

                for (
                    sequences,
                    atac_labels,
                    h3k27ac_targets
                ) in val_loader:

                    sequences = sequences.to(
                        device
                    )

                    atac_labels = atac_labels.to(
                        device
                    )

                    h3k27ac_targets = h3k27ac_targets.to(
                        device
                    )

                    (
                        atac_logits,
                        h3k27ac_output,
                        embeddings
                    ) = model(sequences)

                    atac_logits = atac_logits.squeeze(-1)

                    h3k27ac_output = h3k27ac_output.squeeze(-1)

                    atac_loss = atac_criterion(
                        atac_logits,
                        atac_labels
                    )

                    h3k27ac_loss = h3k27ac_criterion(
                        h3k27ac_output,
                        h3k27ac_targets
                    )

                    total_loss = (
                            atac_loss
                            + lambda_h3k27ac * h3k27ac_loss
                    )

                    val_loss += total_loss.item()

                    atac_probs = torch.sigmoid(
                        atac_logits
                    )

                    all_atac_probs.extend(
                        atac_probs.cpu().numpy()
                    )

                    all_atac_labels.extend(
                        atac_labels.cpu().numpy()
                    )

                    all_h3k27ac_preds.extend(
                        h3k27ac_output.cpu().numpy()
                    )

                    all_h3k27ac_targets.extend(
                        h3k27ac_targets.cpu().numpy()
                    )

            val_auc = roc_auc_score(
                all_atac_labels,
                all_atac_probs
            )

            h3k27ac_pearson, _ = pearsonr(
                all_h3k27ac_preds,
                all_h3k27ac_targets
            )

            scheduler.step(
                val_auc
            )

            avg_train_loss = (
                train_loss
                / len(train_loader)
            )

            avg_val_loss = (
                val_loss
                / len(val_loader)
            )

            print(
                f"Epoch {epoch+1} | "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Val Loss: {avg_val_loss:.4f} | "
                f"ATAC AUROC: {val_auc:.4f} | "
                f"H3K27ac Pearson: "
                f"{h3k27ac_pearson:.4f}"
            )

            mlflow.log_metric(
                "train_loss",
                avg_train_loss,
                step=epoch
            )

            mlflow.log_metric(
                "val_loss",
                avg_val_loss,
                step=epoch
            )

            mlflow.log_metric(
                "ATAC_AUROC",
                val_auc,
                step=epoch
            )

            mlflow.log_metric(
                "H3K27ac_Pearson",
                h3k27ac_pearson,
                step=epoch
            )

            current_score = (
                                    (val_auc / 0.9636)
                                    +
                                    (h3k27ac_pearson / 0.4339)
                            ) / 2

            if current_score > best_score:

                best_score = current_score

                best_auc = val_auc

                best_pearson = h3k27ac_pearson

                patience_counter = 0

                if save_model:

                    torch.save(
                        model.state_dict(),
                        "models/bayesopt_regulatory_resnet.pth"
                    )

                print(
                    f"New best model saved "
                    f"(AUROC={val_auc:.4f})"
                )

            else:

                patience_counter += 1

            if patience_counter >= patience:

                print(
                    "Early stopping triggered."
                )

                break

        mlflow.pytorch.log_model(
            model,
            "regulatory_resnet"
        )

    return (
        best_auc,
        best_pearson
    )

if __name__ == "__main__":

    train_model(
        lr=1e-3,
        weight_decay=1e-4,
        dropout=0.2,
        se_reduction=16,
        lambda_h3k27ac=2.0,
        train_fraction=1.0,
        epochs=25
    )
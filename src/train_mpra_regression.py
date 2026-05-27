import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import mlflow
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.utils.data import random_split

import pandas as pd
import numpy as np

from scipy.stats import pearsonr
from scipy.stats import spearmanr

from src.model import DeepSequenceCNN
from src.mpra_dataset import MPRADataset

def train_mpra_regression():

    mlflow.set_experiment(
        "chromatin_accessibility"
    )

    with mlflow.start_run():
        df = pd.read_csv(
            "./data/processed/mpra_hepg2.csv"
        )

        dataset = MPRADataset(df)

        train_size = int(
            0.8 * len(dataset)
        )

        val_size = len(dataset) - train_size

        epochs = 25
        batch_size = 32
        learning_rate = 1e-3

        train_dataset, val_dataset = random_split(
            dataset,
            [train_size, val_size]
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size
        )

        model = DeepSequenceCNN()

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate
        )

        criterion = nn.MSELoss()

        mlflow.log_param(
            "batch_size",
            batch_size
        )

        mlflow.log_param(
            "learning_rate",
            learning_rate
        )

        mlflow.log_param(
            "epochs",
            epochs
        )

        mlflow.log_param(
            "architecture",
            "deep_cnn"
        )

        print("Train start.")

        best_pearson = 0
        patience = 5
        epochs_without_improvement = 0

        for epoch in range(epochs):

            model.train()

            train_losses = []

            for sequences, targets in train_loader:
                optimizer.zero_grad()

                predictions = model(
                    sequences
                ).squeeze()

                loss = criterion(
                    predictions,
                    targets
                )

                loss.backward()

                optimizer.step()

                train_losses.append(
                    loss.item()
                )

            model.eval()

            val_losses = []

            all_predictions = []
            all_targets = []

            with torch.no_grad():

                for sequences, targets in val_loader:
                    predictions = model(
                        sequences
                    ).squeeze()

                    loss = criterion(
                        predictions,
                        targets
                    )

                    val_losses.append(
                        loss.item()
                    )

                    all_predictions.extend(
                        predictions.numpy()
                    )

                    all_targets.extend(
                        targets.numpy()
                    )

            pearson_corr, _ = pearsonr(
                all_predictions,
                all_targets
            )

            spearman_corr, _ = spearmanr(
                all_predictions,
                all_targets
            )

            mean_train_loss = np.mean(
                train_losses
            )

            mean_val_loss = np.mean(
                val_losses
            )

            print(
                f"Epoch {epoch + 1} | "
                f"Train Loss: {mean_train_loss:.4f} | "
                f"Val Loss: {mean_val_loss:.4f} | "
                f"Pearson: {pearson_corr:.4f} | "
                f"Spearman: {spearman_corr:.4f}"
            )

            if pearson_corr > best_pearson:

                best_pearson = pearson_corr

                epochs_without_improvement = 0

                torch.save(
                    model.state_dict(),
                    "models/best_mpra_model.pth"
                )

                print(
                    f"New best model saved "
                    f"(Pearson={pearson_corr:.4f})"
                )

            else:

                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                print(
                    "Early stopping triggered."
                )

                break

            mlflow.log_metric(
                "train_loss",
                mean_train_loss,
                step=epoch
            )

            mlflow.log_metric(
                "val_loss",
                mean_val_loss,
                step=epoch
            )

            mlflow.log_metric(
                "pearson",
                pearson_corr,
                step=epoch
            )

            mlflow.log_metric(
                "spearman",
                spearman_corr,
                step=epoch
            )

        mlflow.pytorch.log_model(
            model,
            "model"
        )

        plt.figure(figsize=(6, 6))

        plt.scatter(
            all_targets,
            all_predictions,
            alpha=0.3
        )

        plt.xlabel(
            "True MPRA Activity"
        )

        plt.ylabel(
            "Predicted Activity"
        )

        plt.title(
            "MPRA Regression Predictions"
        )

        plt.savefig(
            "./figures/mpra_regression_scatter.png"
        )

        mlflow.log_artifact(
            "./figures/mpra_regression_scatter.png"
        )

        plt.close()

if __name__ == "__main__":

    train_mpra_regression()
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.utils.data import random_split

from src.datasets import SequenceDataset
from src.model import SequenceCNN

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score
)


def train():

    dataset = SequenceDataset(
        "data/processed/liver_accessibility.csv"
    )

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=2,
        shuffle=False
    )

    model = SequenceCNN()

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )

    epochs = 10

    for epoch in range(epochs):

        # -------------------
        # TRAINING
        # -------------------

        model.train()

        total_train_loss = 0

        for batch_x, batch_y in train_loader:

            predictions = model(batch_x)

            loss = criterion(
                predictions,
                batch_y
            )

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            total_train_loss += loss.item()

        average_train_loss = (
            total_train_loss / len(train_loader)
        )

        # -------------------
        # VALIDATION
        # -------------------

        model.eval()

        total_val_loss = 0

        all_predictions = []
        all_labels = []

        with torch.no_grad():

            for batch_x, batch_y in val_loader:
                predictions = model(batch_x)

                probabilities = torch.sigmoid(
                    predictions
                )

                loss = criterion(
                    predictions,
                    batch_y
                )

                total_val_loss += loss.item()

                all_predictions.extend(
                    probabilities.view(-1).tolist()
                )

                all_labels.extend(
                    batch_y.view(-1).tolist()
                )

        average_val_loss = (
                total_val_loss / len(val_loader)
        )

        accuracy = accuracy_score(
            all_labels,
            [p > 0.5 for p in all_predictions]
        )

        roc_auc = roc_auc_score(
            all_labels,
            all_predictions
        )

        print(
            f"Epoch {epoch + 1} | "
            f"Train Loss: {average_train_loss:.4f} | "
            f"Val Loss: {average_val_loss:.4f} | "
            f"Accuracy: {accuracy:.4f} | "
            f"ROC-AUC: {roc_auc:.4f}"
        )

    torch.save(
        model.state_dict(),
        "./models/liver_accessibility_cnn.pth"
    )

if __name__ == "__main__":

    train()

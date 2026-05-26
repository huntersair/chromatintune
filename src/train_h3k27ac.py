import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.utils.data import Subset

from src.datasets import SequenceDataset
from src.model_h3k27ac import H3K27acModel

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score
)

def train():

    dataset = SequenceDataset(
        "data/processed/liver_accessibility_multimodal.csv"
    )

    train_chroms = [
        "chr1",
        "chr2"
    ]

    val_chroms = [
        "chr3"
    ]

    train_indices = []
    val_indices = []

    for idx in range(len(dataset)):

        _, _, _, chrom = dataset[idx]

        if chrom in train_chroms:
            train_indices.append(idx)

        elif chrom in val_chroms:
            val_indices.append(idx)

    train_dataset = Subset(
        dataset,
        train_indices
    )

    val_dataset = Subset(
        dataset,
        val_indices
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False
    )

    model = H3K27acModel()

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )

    epochs = 10

    print("Train start.")

    for epoch in range(epochs):

        # -------------------
        # TRAINING
        # -------------------

        model.train()

        total_train_loss = 0

        for _, batch_h3k27ac, batch_y, _ in train_loader:

            predictions = model(
                batch_h3k27ac
            )

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

            for _, batch_h3k27ac, batch_y, _ in val_loader:

                predictions = model(
                    batch_h3k27ac
                )

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

    print("Model saved.")

if __name__ == "__main__":

    train()

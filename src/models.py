"""
models.py – Definisi model/pipeline untuk kedua arsitektur.

Pipeline 1 (Non-Transformer):
  - TF-IDF + ResNet50 frozen → concat (L2 norm) → MLP classifier
  
Pipeline 2 (Transformer):
  - XLM-RoBERTa + ViT frozen → concat → FusionHead (MLP)
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Optional

from src.config import NUM_CLASSES, HIDDEN_DIM, DROPOUT


# ═══════════════════════════════════════════════════════════════════════════
#  FUSION HEAD (dipakai oleh kedua pipeline)
# ═══════════════════════════════════════════════════════════════════════════

class FusionHead(nn.Module):
    """
    MLP klasifikasi di atas fitur gabungan (concat) dari dua modalitas.
    
    Lebih baik dari Logistic Regression linear karena bisa belajar bobot
    relatif tiap modalitas secara otomatis via hidden layer.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = HIDDEN_DIM,
        n_classes: int = NUM_CLASSES,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ═══════════════════════════════════════════════════════════════════════════
#  TRAINING UTILITIES
# ═══════════════════════════════════════════════════════════════════════════

class EarlyStopping:
    """Early stopping berdasarkan validation loss."""

    def __init__(self, patience: int = 5, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.should_stop = False

    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


def train_fusion_head(
    model: FusionHead,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    num_epochs: int = 30,
    batch_size: int = 32,
    lr: float = 1e-3,
    patience: int = 5,
    class_weights: Optional[np.ndarray] = None,
    device: Optional[torch.device] = None,
):
    """
    Training loop untuk FusionHead dengan early stopping.
    
    Returns:
        model: Model terlatih (best state).
        history: Dict berisi train_loss, val_loss, val_f1 per epoch.
    """
    from sklearn.metrics import f1_score

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    # Class weights untuk handle ketidakseimbangan kelas
    if class_weights is not None:
        weight_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )
    early_stop = EarlyStopping(patience=patience)

    # Convert numpy ke tensor
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.long).to(device)

    train_ds = torch.utils.data.TensorDataset(X_train_t, y_train_t)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True
    )

    best_state = None
    best_val_f1 = 0.0
    history = {"train_loss": [], "val_loss": [], "val_f1": []}

    for epoch in range(num_epochs):
        # ── Train ──
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(xb)
        epoch_loss /= len(train_ds)

        # ── Validate ──
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t)
            val_loss = criterion(val_logits, y_val_t).item()
            val_preds = val_logits.argmax(dim=1).cpu().numpy()
            val_f1 = f1_score(y_val, val_preds, average="macro")

        history["train_loss"].append(epoch_loss)
        history["val_loss"].append(val_loss)
        history["val_f1"].append(val_f1)

        scheduler.step(val_loss)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"Epoch {epoch+1:3d}/{num_epochs} | "
                f"Train Loss: {epoch_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Macro-F1: {val_f1:.4f}"
            )

        if early_stop(val_loss):
            print(f"Early stopping at epoch {epoch+1}")
            break

    # Restore best state
    if best_state is not None:
        model.load_state_dict(best_state)
    model = model.to(device)
    model.eval()

    print(f"\nBest Validation Macro-F1: {best_val_f1:.4f}")
    return model, history


def predict_with_fusion_head(
    model: FusionHead,
    X: np.ndarray,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """Prediksi kelas dari fitur gabungan."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    X_t = torch.tensor(X, dtype=torch.float32).to(device)
    with torch.no_grad():
        logits = model(X_t)
        preds = logits.argmax(dim=1).cpu().numpy()
    return preds


def predict_proba_with_fusion_head(
    model: FusionHead,
    X: np.ndarray,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """Prediksi probabilitas dari fitur gabungan."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    X_t = torch.tensor(X, dtype=torch.float32).to(device)
    with torch.no_grad():
        logits = model(X_t)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
    return probs

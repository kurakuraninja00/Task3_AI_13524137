"""
evaluation.py – Metrik evaluasi, confusion matrix, error analysis.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Dict

from sklearn.metrics import (
    f1_score,
    classification_report,
    confusion_matrix,
)
from src.config import CLASSES, ID2LABEL


def compute_macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Hitung macro F1-score (metrik evaluasi kompetisi)."""
    return f1_score(y_true, y_pred, average="macro")


def print_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: List[str] = CLASSES,
) -> str:
    """Print dan return classification report."""
    report = classification_report(y_true, y_pred, target_names=target_names)
    print(report)
    return report


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: List[str] = CLASSES,
    title: str = "Confusion Matrix",
    figsize: tuple = (8, 6),
    normalize: bool = False,
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """
    Plot confusion matrix dengan heatmap.
    
    Baris = label asli, kolom = prediksi.
    """
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
        fmt = ".2f"
    else:
        fmt = "d"

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_training_history(
    history: Dict[str, List[float]],
    figsize: tuple = (14, 5),
) -> plt.Figure:
    """Plot training history: loss dan F1 per epoch."""
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Loss
    axes[0].plot(history["train_loss"], label="Train Loss", linewidth=2)
    axes[0].plot(history["val_loss"], label="Val Loss", linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training & Validation Loss", fontweight="bold")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # F1
    axes[1].plot(history["val_f1"], label="Val Macro-F1", color="green", linewidth=2)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Macro F1")
    axes[1].set_title("Validation Macro F1-Score", fontweight="bold")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def analyze_errors(
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_samples: int = 10,
) -> pd.DataFrame:
    """
    Analisis contoh prediksi yang salah.
    
    Returns:
        DataFrame dengan kolom: character_name, personality (truncated),
        true_label, predicted_label
    """
    errors_mask = y_true != y_pred
    error_df = df[errors_mask].copy()
    error_df["true_label"] = [ID2LABEL[y] for y in y_true[errors_mask]]
    error_df["predicted_label"] = [ID2LABEL[y] for y in y_pred[errors_mask]]
    error_df["personality_short"] = error_df["personality"].str[:150] + "..."

    cols = ["character_name", "true_label", "predicted_label", "personality_short"]
    return error_df[cols].head(n_samples)


def compare_pipelines(
    results: Dict[str, Dict],
    figsize: tuple = (10, 6),
) -> plt.Figure:
    """
    Bar chart perbandingan macro F1 per kelas untuk beberapa pipeline.
    
    Args:
        results: {pipeline_name: {"per_class_f1": [f1_deredere, f1_kuudere, f1_tsundere], "macro_f1": float}}
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    x = np.arange(len(CLASSES))
    width = 0.35
    n_pipelines = len(results)
    
    for i, (name, res) in enumerate(results.items()):
        offset = (i - (n_pipelines - 1) / 2) * width
        bars = ax.bar(x + offset, res["per_class_f1"], width, label=f"{name} (macro={res['macro_f1']:.3f})")
        # Annotate
        for bar, val in zip(bars, res["per_class_f1"]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    
    ax.set_xticks(x)
    ax.set_xticklabels(CLASSES)
    ax.set_ylabel("F1-Score")
    ax.set_title("Per-Class F1 Comparison Across Pipelines", fontweight="bold")
    ax.legend()
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    return fig

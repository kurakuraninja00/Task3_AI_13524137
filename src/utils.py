"""
utils.py – Fungsi utilitas umum: seed, device detection, data loading.
"""
import random
import numpy as np
import pandas as pd
import torch
from pathlib import Path

from src.config import SEED, TRAIN_CSV, TEST_CSV, TRAIN_DIR, TEST_DIR, LABEL2ID


def set_seed(seed: int = SEED) -> None:
    """Set seed untuk reprodusibilitas penuh."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Pilih GPU jika tersedia, lalu CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_train_df() -> pd.DataFrame:
    """Load train.csv, tambah kolom path absolut dan label numerik."""
    df = pd.read_csv(TRAIN_CSV)
    # file_name di train.csv sudah termasuk subfolder kelas, misal "deredere/xxx.webp"
    df["image_path"] = df["file_name"].apply(lambda f: str(TRAIN_DIR / f))
    df["label_id"] = df["label"].map(LABEL2ID)
    return df


def load_test_df() -> pd.DataFrame:
    """Load test.csv, tambah kolom path absolut."""
    df = pd.read_csv(TEST_CSV)
    # file_name di test.csv flat tanpa subfolder, misal "xxx.webp"
    df["image_path"] = df["file_name"].apply(lambda f: str(TEST_DIR / f))
    return df

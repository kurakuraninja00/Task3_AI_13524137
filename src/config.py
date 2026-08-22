"""
config.py – Konstanta global, path dataset, hyperparameter default.
"""
import os
from pathlib import Path

# ─── Path ────────────────────────────────────────────────────────────────
# Otomatis detect apakah running di Kaggle atau lokal
IS_KAGGLE = os.environ.get("KAGGLE_KERNEL_RUN_TYPE") is not None

if IS_KAGGLE:
    PROJECT_ROOT = Path("/kaggle/working")
    DATASET_ROOT = Path("/kaggle/input/dere-detector")
else:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    DATASET_ROOT = PROJECT_ROOT / "dataset"

TRAIN_DIR    = DATASET_ROOT / "train"
TEST_DIR     = DATASET_ROOT / "test"
TRAIN_CSV    = DATASET_ROOT / "train.csv"
TEST_CSV     = DATASET_ROOT / "test.csv"
SAMPLE_SUB   = DATASET_ROOT / "sample_submission.csv"

# ─── Label ───────────────────────────────────────────────────────────────
CLASSES      = ["deredere", "kuudere", "tsundere"]
NUM_CLASSES  = len(CLASSES)
LABEL2ID     = {c: i for i, c in enumerate(CLASSES)}
ID2LABEL     = {i: c for i, c in enumerate(CLASSES)}

# ─── Reproducibility ────────────────────────────────────────────────────
SEED = 42

# ─── Image ───────────────────────────────────────────────────────────────
IMG_SIZE         = 224
IMG_MEAN         = [0.485, 0.456, 0.406]   # ImageNet stats
IMG_STD          = [0.229, 0.224, 0.225]

# ─── Text ────────────────────────────────────────────────────────────────
TFIDF_MAX_FEATURES = 4000
TFIDF_NGRAM_RANGE  = (1, 2)
MAX_SEQ_LEN        = 512                   # Untuk BERT/XLM-R

# ─── Model ───────────────────────────────────────────────────────────────
CNN_BACKBONE   = "resnet50"
VIT_MODEL_NAME = "google/vit-base-patch16-224-in21k"
TXT_MODEL_NAME = "xlm-roberta-base"

# ─── Training ────────────────────────────────────────────────────────────
VAL_SIZE       = 0.2
BATCH_SIZE     = 32
LEARNING_RATE  = 1e-3
NUM_EPOCHS     = 30
PATIENCE       = 5   # early stopping
DROPOUT        = 0.5
HIDDEN_DIM     = 128

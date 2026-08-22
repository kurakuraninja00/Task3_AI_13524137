"""
features.py – Ekstraksi fitur untuk kedua pipeline.

Pipeline 1 (Non-Transformer):
  - TF-IDF (1–2 gram) untuk teks
  - Histogram warna + HOG untuk gambar (baseline klasik)
  - ResNet50 pretrained (backbone dibekukan) untuk gambar (deep)

Pipeline 2 (Transformer):
  - XLM-RoBERTa CLS embedding untuk teks
  - ViT CLS embedding untuk gambar
"""
import numpy as np
from typing import List, Optional
from tqdm.auto import tqdm

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from skimage.feature import hog
from skimage.color import rgb2gray

from src.config import (
    TFIDF_MAX_FEATURES, TFIDF_NGRAM_RANGE, IMG_SIZE, IMG_MEAN, IMG_STD,
    CNN_BACKBONE, VIT_MODEL_NAME, TXT_MODEL_NAME, MAX_SEQ_LEN,
)
from src.preprocessing import load_and_preprocess_image, clean_personality_text


# ═══════════════════════════════════════════════════════════════════════════
#  TEKS — KLASIK
# ═══════════════════════════════════════════════════════════════════════════

def build_tfidf_vectorizer(texts: List[str]) -> TfidfVectorizer:
    """Fit TF-IDF vectorizer pada daftar teks."""
    vec = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        ngram_range=TFIDF_NGRAM_RANGE,
        sublinear_tf=True,
        strip_accents="unicode",
    )
    vec.fit(texts)
    return vec


# ═══════════════════════════════════════════════════════════════════════════
#  GAMBAR — KLASIK (Histogram + HOG)
# ═══════════════════════════════════════════════════════════════════════════

def extract_color_histogram(img_array: np.ndarray, bins: int = 8) -> np.ndarray:
    """Histogram warna 3-channel, dinormalisasi."""
    hists = []
    for ch in range(3):
        h, _ = np.histogram(img_array[:, :, ch], bins=bins, range=(0, 256))
        hists.append(h)
    feat = np.concatenate(hists).astype(np.float64)
    feat /= feat.sum() + 1e-8
    return feat


def extract_hog_features(img_array: np.ndarray, resize: int = 96) -> np.ndarray:
    """HOG features dari gambar grayscale yang di-resize."""
    from skimage.transform import resize as sk_resize
    gray = rgb2gray(img_array)
    gray_resized = sk_resize(gray, (resize, resize), anti_aliasing=True)
    features = hog(
        gray_resized,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        feature_vector=True,
    )
    return features


def extract_classical_image_features(img_array: np.ndarray) -> np.ndarray:
    """Gabung histogram warna + HOG jadi 1 vektor fitur."""
    hist = extract_color_histogram(img_array)
    hog_feat = extract_hog_features(img_array)
    return np.concatenate([hist, hog_feat])


# ═══════════════════════════════════════════════════════════════════════════
#  GAMBAR — CNN PRETRAINED (ResNet50)
# ═══════════════════════════════════════════════════════════════════════════

def get_cnn_model():
    """
    Load ResNet50 pretrained, buang FC layer terakhir → embedding 2048-d.
    Backbone dibekukan (requires_grad=False).
    """
    import torch
    import torch.nn as nn
    import torchvision.models as tvm

    weights = tvm.ResNet50_Weights.IMAGENET1K_V2
    cnn = tvm.resnet50(weights=weights)
    cnn.fc = nn.Identity()  # Output embedding 2048-d, bukan logit 1000-kelas
    for p in cnn.parameters():
        p.requires_grad = False
    cnn.eval()
    return cnn


def get_cnn_transforms():
    """Transform untuk ResNet50 (ImageNet normalization)."""
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMG_MEAN, std=IMG_STD),
    ])


def embed_images_cnn(image_paths: List[str], device=None, batch_size: int = 32) -> np.ndarray:
    """
    Embed batch gambar menggunakan ResNet50 pretrained.
    Returns: array (N, 2048).
    """
    import torch

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cnn = get_cnn_model().to(device)
    tfms = get_cnn_transforms()
    embeddings = []

    for i in tqdm(range(0, len(image_paths), batch_size), desc="CNN embedding"):
        batch_paths = image_paths[i:i + batch_size]
        imgs = []
        for p in batch_paths:
            img = load_and_preprocess_image(p)
            imgs.append(tfms(img))
        batch_tensor = torch.stack(imgs).to(device)
        with torch.no_grad():
            emb = cnn(batch_tensor).cpu().numpy()
        embeddings.append(emb)

    return np.vstack(embeddings)


# ═══════════════════════════════════════════════════════════════════════════
#  TEKS — TRANSFORMER (XLM-RoBERTa)
# ═══════════════════════════════════════════════════════════════════════════

def get_text_model_and_tokenizer(model_name: str = TXT_MODEL_NAME):
    """Load XLM-RoBERTa (atau model lain) + tokenizer."""
    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, tokenizer


def embed_texts_transformer(
    texts: List[str],
    model=None,
    tokenizer=None,
    model_name: str = TXT_MODEL_NAME,
    device=None,
    batch_size: int = 16,
    max_length: int = MAX_SEQ_LEN,
) -> np.ndarray:
    """
    Embed batch teks menggunakan transformer (CLS token).
    Returns: array (N, hidden_dim).
    """
    import torch

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model is None or tokenizer is None:
        model, tokenizer = get_text_model_and_tokenizer(model_name)
    model = model.to(device)

    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Text embedding"):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True,
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        cls_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        embeddings.append(cls_emb)

    return np.vstack(embeddings)


# ═══════════════════════════════════════════════════════════════════════════
#  GAMBAR — TRANSFORMER (ViT)
# ═══════════════════════════════════════════════════════════════════════════

def get_vit_model_and_processor(model_name: str = VIT_MODEL_NAME):
    """Load ViT + processor."""
    from transformers import ViTModel, ViTImageProcessor
    processor = ViTImageProcessor.from_pretrained(model_name)
    model = ViTModel.from_pretrained(model_name)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, processor


def embed_images_vit(
    image_paths: List[str],
    model=None,
    processor=None,
    model_name: str = VIT_MODEL_NAME,
    device=None,
    batch_size: int = 16,
) -> np.ndarray:
    """
    Embed batch gambar menggunakan ViT pretrained (CLS token).
    Returns: array (N, 768).
    """
    import torch

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model is None or processor is None:
        model, processor = get_vit_model_and_processor(model_name)
    model = model.to(device)

    embeddings = []
    for i in tqdm(range(0, len(image_paths), batch_size), desc="ViT embedding"):
        batch_paths = image_paths[i:i + batch_size]
        imgs = [load_and_preprocess_image(p) for p in batch_paths]
        inputs = processor(images=imgs, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        cls_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        embeddings.append(cls_emb)

    return np.vstack(embeddings)


# ═══════════════════════════════════════════════════════════════════════════
#  FUSION
# ═══════════════════════════════════════════════════════════════════════════

def build_fusion_features(
    text_matrix: np.ndarray,
    image_matrix: np.ndarray,
) -> np.ndarray:
    """
    Gabung fitur teks + gambar dengan L2-normalisasi per blok terlebih dahulu.
    Ini mencegah satu modalitas mendominasi yang lain karena perbedaan skala.
    """
    text_n = normalize(text_matrix)
    img_n = normalize(image_matrix)
    return np.hstack([text_n, img_n])

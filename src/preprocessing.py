"""
preprocessing.py – Pembersihan teks dan pemrosesan gambar.

Menangani:
  - Escape sequence literal (\\uXXXX, \\n) dari teks yang di-paste dari JSON/PDF
  - Strikethrough markdown, whitespace berlebih
  - Konversi RGBA → RGB dengan background putih (341/1009 gambar RGBA)
  - Resize ke ukuran standar untuk model
"""
import re
from PIL import Image
from src.config import IMG_SIZE


# ═══════════════════════════════════════════════════════════════════════════
# TEKS
# ═══════════════════════════════════════════════════════════════════════════

def clean_personality_text(text: str) -> str:
    """
    Bersihkan teks personality dari artefak encoding/markup.
    
    Kasus yang ditangani:
      - \\uXXXX literal  → karakter Unicode sebenarnya
      - \\n literal       → newline
      - ~~strikethrough~~ → teks tanpa markup
      - Whitespace berlebih
    
    Penting untuk test case real-world (Aldoy & user) yang mengandung
    escape sequence dari PDF.
    """
    if not isinstance(text, str):
        return ""
    # Tangani \\uXXXX literal (sering dari JSON/markdown mentah)
    text = re.sub(
        r"\\u([0-9a-fA-F]{4})",
        lambda m: chr(int(m.group(1), 16)),
        text,
    )
    # Tangani \\n literal → newline
    text = text.replace("\\n", "\n")
    # Buang syntax strikethrough, teks tetap dipakai
    text = re.sub(r"~~(.*?)~~", r"\1", text)
    # Normalisasi whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


# ═══════════════════════════════════════════════════════════════════════════
# GAMBAR
# ═══════════════════════════════════════════════════════════════════════════

def load_and_preprocess_image(path: str, size: int = IMG_SIZE) -> Image.Image:
    """
    Load gambar, handle RGBA → RGB (composite ke background putih), resize.
    
    Args:
        path: Path ke file gambar.
        size: Target width & height (square).
    
    Returns:
        PIL Image mode RGB, ukuran (size, size).
    """
    with Image.open(path) as im:
        # Konversi ke RGBA dulu untuk handle semua mode (termasuk P, LA, dll.)
        im = im.convert("RGBA")
        # Composite ke background putih → buang alpha channel
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        im = Image.alpha_composite(bg, im).convert("RGB")
        im = im.resize((size, size), Image.LANCZOS)
    return im


def load_image_as_array(path: str, size: int = IMG_SIZE):
    """Load gambar sebagai numpy array (H, W, 3), dtype uint8."""
    import numpy as np
    img = load_and_preprocess_image(path, size)
    return np.array(img, dtype=np.uint8)

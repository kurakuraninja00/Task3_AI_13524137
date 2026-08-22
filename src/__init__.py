"""
DereDetector – Multimodal Dere Archetype Classification
=======================================================
Paket utilitas yang berisi seluruh logika preprocessing, feature engineering,
model pipeline, evaluasi, dan explainability untuk klasifikasi 3-kelas
(deredere / kuudere / tsundere) dari gambar + teks kepribadian karakter anime.

Modul:
    - config        : Konstanta dan path konfigurasi
    - preprocessing : Pembersihan teks & pemrosesan gambar
    - features      : Ekstraksi fitur (TF-IDF, HOG, CNN, ViT, BERT)
    - models        : Definisi pipeline non-transformer & transformer
    - evaluation    : Metrik, confusion matrix, error analysis
    - explainability: Grad-CAM, LIME, attention rollout
    - utils         : Helper umum (seed, device, dll.)
"""

__version__ = "1.0.0"

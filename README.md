# DereDetector: Multimodal Dere Archetype Classification

Repositori ini memuat *source code* dan dokumentasi analisis untuk kompetisi klasifikasi *Dere Archetype* (Kuudere, Tsundere, Deredere) menggunakan dataset multimodal (Teks Bio dan Gambar Karakter).

## ✨ Fitur Utama
1. **Multimodal Fusion Pipeline**: Menggunakan gabungan representasi Visual dan Tekstual secara seimbang via *L2 Normalization* & PCA.
2. **Pipeline 1 (Baseline Kuat)**: ResNet50 (Visual) + TF-IDF (Text).
3. **Pipeline 2 (Transformer-based)**: ViT (Visual) + XLM-RoBERTa (Text) dengan *Mean-Pooling*.
4. **Regularisasi Ekstensif**: 5-Fold Stratified Cross Validation, Test-Time Augmentation (TTA), PCA Bottleneck, AdamW Decoupled Weight Decay, Dropout 0.5 untuk mengatasi kondisi data latih yang sangat terbatas (707 sampel).
5. **Soft-Voting Ensemble**: Prediksi akhir menggabungkan probabilitas P1 dan P2 untuk menekan *noise* (bias varians).

## 📂 Struktur Repositori Terkini
- `notebooks/v5.ipynb`: Notebook eksperimen terlengkap, paling optimal, bebas *overfitting*, dan siap di-*submit* ke platform Kaggle.
- `inference.md`: Panduan tata cara menjalankan inferensi di lokal (menggunakan model *Ensemble* hasil pelatihan) dengan format skrip Python.
- `src/`: Berisi skrip modular jika ingin mengembangkan fitur tanpa Notebook.
- `solution.md` & `solution2.md`: Laporan evaluasi awal mengenai diagnosis *overfitting* beserta solusi penanganannya secara matematis dan logis.

## 🚀 Cara Reproduksi (Training)
1. Unggah `notebooks/v5.ipynb` ke *environment* Kaggle Anda.
2. Pastikan dataset kompetisi `dere-detector.zip` terhubung pada kernel.
3. Jalankan `Run All` untuk memulai ekstraksi fitur, proses latih K-Fold, evaluasi, hingga proses ensemble *soft-voting*.
4. Setelah selesai, unduh file `submission_models.zip` dari panel *Output* (berisi bobot hasil training untuk *deployment* lokal).

## 🔮 Panduan Inferensi
Untuk panduan detil mengenai *deployment* dan cara memanggil model secara lokal, baca dokumen [inference.md](./inference.md).
# DereDetector: Panduan Inferensi Lokal

Setelah melatih model di Kaggle dan mengunduh `submission_models.zip`, Anda dapat menggunakan bobot model (*weights*) tersebut di komputer lokal Anda untuk melakukan prediksi pada karakter baru.

## 1. Persiapan Lingkungan

Pastikan Anda memiliki *library* berikut terinstal di *environment* Python Anda:
```bash
pip install torch torchvision transformers scikit-learn numpy pillow joblib
```

## 2. Struktur File Model

Ekstrak `submission_models.zip`. Anda akan mendapatkan file-file berikut:
- `model_p1.pth` (Bobot PyTorch untuk Pipeline 1)
- `model_p2.pth` (Bobot PyTorch untuk Pipeline 2)
- `pca_txt.pkl` (Model PCA untuk dimensi teks)
- `pca_img.pkl` (Model PCA untuk dimensi gambar)
- `tfidf_vec.pkl` (Model TF-IDF Vectorizer)

Letakkan kelima file ini di sebuah folder, misalnya `models/`.

## 3. Skrip Inferensi (Ensemble)

Berikut adalah contoh skrip Python untuk melakukan inferensi menggunakan *Soft Voting Ensemble* dari Pipeline 1 (ResNet + TF-IDF) dan Pipeline 2 (ViT + XLM-R).

```python
import torch
import joblib
import numpy as np
from PIL import Image
from transformers import AutoTokenizer, AutoModel, ViTModel, ViTImageProcessor
import torchvision.models as tvm
from torchvision import transforms

# ==========================================
# 1. LOAD KOMPONEN NON-DEEP-LEARNING
# ==========================================
print("Loading PCA & TF-IDF...")
pca_txt_p1 = joblib.load('models/pca_txt.pkl')
pca_img_p1 = joblib.load('models/pca_img.pkl')
pca_txt_p2 = joblib.load('models/pca_txt_p2.pkl')
pca_img_p2 = joblib.load('models/pca_img_p2.pkl')
tfidf_vec = joblib.load('models/tfidf_vec.pkl')

# ==========================================
# 2. LOAD BACKBONE MODELS (FROZEN)
# ==========================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("Loading Image Extractors (ResNet & ViT)...")
cnn = tvm.resnet50(weights=None)
cnn.fc = torch.nn.Identity()
cnn.eval().to(device)
cnn_tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

vit_proc = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
vit = ViTModel.from_pretrained('google/vit-base-patch16-224').eval().to(device)

print("Loading Text Extractor (XLM-R)...")
xlmr_tok = AutoTokenizer.from_pretrained('xlm-roberta-base')
xlmr = AutoModel.from_pretrained('xlm-roberta-base').eval().to(device)

# ==========================================
# 3. LOAD FUSION HEAD MODELS (MLP)
# ==========================================
import torch.nn as nn

class FusionHead(nn.Module):
    def __init__(self, input_dim=200, hidden_dim=64, num_classes=3, dropout=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    def forward(self, x): return self.net(x)

print("Loading Fusion Weights...")
model_p1 = FusionHead().to(device)
model_p1.load_state_dict(torch.load('models/model_p1.pth', map_location=device))
model_p1.eval()

model_p2 = FusionHead().to(device)
model_p2.load_state_dict(torch.load('models/model_p2.pth', map_location=device))
model_p2.eval()

CLASSES = ['deredere', 'kuudere', 'tsundere']

# ==========================================
# 4. FUNGSI PREDIKSI
# ==========================================
def predict_dere(text, image_path):
    # A. Ekstraksi Gambar
    im = Image.open(image_path).convert("RGBA")
    bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
    im = Image.alpha_composite(bg, im).convert("RGB")
    
    with torch.no_grad():
        # ResNet
        cnn_in = cnn_tfms(im).unsqueeze(0).to(device)
        img_p1 = cnn(cnn_in).cpu().numpy()
        # ViT
        vit_in = vit_proc(images=[np.array(im)], return_tensors="pt").to(device)
        img_p2 = vit(**vit_in).last_hidden_state[:, 0, :].cpu().numpy()
        
    # B. Ekstraksi Teks
    txt_p1 = tfidf_vec.transform([text]).toarray()
    with torch.no_grad():
        # XLM-R Mean Pooling
        xlm_in = xlmr_tok([text], return_tensors="pt", truncation=True, max_length=128).to(device)
        out = xlmr(**xlm_in)
        mask = xlm_in['attention_mask'].unsqueeze(-1).float()
        txt_p2 = ((out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)).cpu().numpy()
        
    # C. PCA & L2 Normalization (Fusion)
    def normalize(v): return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-8)
    
    # P1 Fusion
    txt_p1_pca = pca_txt_p1.transform(txt_p1)
    img_p1_pca = pca_img_p1.transform(img_p1)
    feat_p1 = np.hstack([normalize(txt_p1_pca), normalize(img_p1_pca)])
    
    # P2 Fusion
    txt_p2_pca = pca_txt_p2.transform(txt_p2)
    img_p2_pca = pca_img_p2.transform(img_p2)
    feat_p2 = np.hstack([normalize(txt_p2_pca), normalize(img_p2_pca)])
    
    # D. Inference
    with torch.no_grad():
        prob_p1 = torch.softmax(model_p1(torch.tensor(feat_p1, dtype=torch.float32).to(device)), dim=1).cpu().numpy()
        prob_p2 = torch.softmax(model_p2(torch.tensor(feat_p2, dtype=torch.float32).to(device)), dim=1).cpu().numpy()
        
    # Soft Voting
    prob_final = (prob_p1 + prob_p2) / 2
    pred_idx = np.argmax(prob_final, axis=1)[0]
    
    return CLASSES[pred_idx], prob_final[0]

# ==========================================
# 5. TESTING
# ==========================================
if __name__ == "__main__":
    bio = "I act like I don't care, but I secretly do!"
    img = "aldoy.png" # Ganti dengan gambar karakter
    
    pred, probs = predict_dere(bio, img)
    print(f"\nPrediksi Final: {pred.upper()}")
    for c, p in zip(CLASSES, probs):
        print(f"{c}: {p*100:.1f}%")
```

## Penjelasan Singkat
- Model tidak memerlukan arsitektur besar untuk dijalankan, karena MLP kita sangat ringan (hanya berukuran beberapa KB).
- Kebutuhan memori terbesar adalah mengunggah bobot pretrained standar (`xlm-roberta`, `resnet50`, `vit-base`) dari HuggingFace dan TorchVision yang akan secara otomatis di-download ke cache lokal Anda saat skrip pertama kali dieksekusi.

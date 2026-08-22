"""
explainability.py – Grad-CAM, LIME, dan attention visualization.

Memberikan insight tentang *mengapa* model memprediksi kelas tertentu:
  - Grad-CAM: area gambar mana yang berkontribusi (rambut? mata? ekspresi?)
  - LIME (teks): kata-kata mana yang mendorong prediksi ("cold", "playful", "blush")
  - Attention Rollout: visualisasi attention ViT
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, List, Optional
from PIL import Image

from src.config import CLASSES


# ═══════════════════════════════════════════════════════════════════════════
#  GRAD-CAM untuk CNN
# ═══════════════════════════════════════════════════════════════════════════

def compute_gradcam(
    model,
    input_tensor,
    target_class: int,
    target_layer,
) -> np.ndarray:
    """
    Compute Grad-CAM heatmap untuk CNN model.
    
    Args:
        model: Full model (backbone + classifier)
        input_tensor: Input gambar (1, C, H, W)
        target_class: Index kelas target
        target_layer: Layer CNN untuk generate heatmap (misal model.layer4[-1])
    
    Returns:
        Heatmap array (H, W) dinormalisasi ke [0, 1]
    """
    import torch
    import torch.nn.functional as F

    activations = {}
    gradients = {}

    def forward_hook(module, input, output):
        activations["value"] = output.detach()

    def backward_hook(module, grad_input, grad_output):
        gradients["value"] = grad_output[0].detach()

    fh = target_layer.register_forward_hook(forward_hook)
    bh = target_layer.register_full_backward_hook(backward_hook)

    model.eval()
    output = model(input_tensor)

    model.zero_grad()
    one_hot = torch.zeros_like(output)
    one_hot[0, target_class] = 1
    output.backward(gradient=one_hot)

    # Pooling gradien → bobot per channel
    weights = gradients["value"].mean(dim=[2, 3], keepdim=True)
    cam = (weights * activations["value"]).sum(dim=1, keepdim=True)
    cam = F.relu(cam)
    cam = cam.squeeze().cpu().numpy()

    # Normalisasi
    cam = cam - cam.min()
    if cam.max() > 0:
        cam = cam / cam.max()

    fh.remove()
    bh.remove()

    return cam


def visualize_gradcam(
    image: Image.Image,
    heatmap: np.ndarray,
    title: str = "Grad-CAM",
    alpha: float = 0.4,
    figsize: tuple = (10, 5),
) -> plt.Figure:
    """Overlay Grad-CAM heatmap pada gambar original."""
    from skimage.transform import resize as sk_resize

    img_array = np.array(image)
    heatmap_resized = sk_resize(heatmap, (img_array.shape[0], img_array.shape[1]))

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    axes[0].imshow(img_array)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(heatmap_resized, cmap="jet")
    axes[1].set_title("Grad-CAM Heatmap")
    axes[1].axis("off")

    axes[2].imshow(img_array)
    axes[2].imshow(heatmap_resized, cmap="jet", alpha=alpha)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════
#  LIME untuk Teks
# ═══════════════════════════════════════════════════════════════════════════

def explain_text_with_lime(
    text: str,
    predict_proba_fn: Callable,
    class_names: List[str] = CLASSES,
    num_features: int = 15,
    num_samples: int = 500,
):
    """
    Jelaskan prediksi teks menggunakan LIME.
    
    Args:
        text: Teks personality yang akan dijelaskan.
        predict_proba_fn: Fungsi yang menerima list of strings, return array (N, num_classes).
        class_names: Nama kelas.
        num_features: Jumlah fitur teratas yang ditampilkan.
        num_samples: Jumlah perturbasi untuk LIME.
    
    Returns:
        LIME Explanation object.
    """
    from lime.lime_text import LimeTextExplainer

    explainer = LimeTextExplainer(class_names=class_names)
    explanation = explainer.explain_instance(
        text,
        predict_proba_fn,
        num_features=num_features,
        num_samples=num_samples,
    )
    return explanation


def plot_lime_explanation(
    explanation,
    label: int = None,
    title: str = "LIME Text Explanation",
    figsize: tuple = (10, 6),
) -> plt.Figure:
    """
    Plot LIME explanation sebagai bar chart horizontal.
    """
    if label is None:
        label = explanation.available_labels()[0]

    exp_list = explanation.as_list(label=label)
    words = [w for w, s in exp_list]
    scores = [s for w, s in exp_list]

    fig, ax = plt.subplots(figsize=figsize)
    colors = ["green" if s > 0 else "red" for s in scores]
    ax.barh(words, scores, color=colors)
    ax.set_xlabel("Contribution Score")
    ax.set_title(f"{title} (predicted: {CLASSES[label]})", fontweight="bold")
    ax.invert_yaxis()
    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════
#  ATTENTION ROLLOUT untuk ViT
# ═══════════════════════════════════════════════════════════════════════════

def attention_rollout(attentions: list, discard_ratio: float = 0.0) -> np.ndarray:
    """
    Compute attention rollout dari list attention matrices.
    
    Args:
        attentions: List of attention tensors dari ViT, masing-masing (B, H, S, S)
        discard_ratio: Proporsi attention terendah yang di-zero-kan (noise reduction)
    
    Returns:
        Rollout matrix (S, S)
    """
    import torch

    result = torch.eye(attentions[0].size(-1))
    with torch.no_grad():
        for attention in attentions:
            # Rata-rata antar heads
            attention_heads_fused = attention.mean(dim=1).squeeze(0)

            # Discard lowest attention
            if discard_ratio > 0:
                flat = attention_heads_fused.view(-1)
                _, indices = flat.topk(int(flat.size(0) * discard_ratio), largest=False)
                flat[indices] = 0

            # Re-normalize
            attention_heads_fused = attention_heads_fused / attention_heads_fused.sum(dim=-1, keepdim=True)

            # Tambah residual connection (identity)
            I = torch.eye(attention_heads_fused.size(-1))
            a = (attention_heads_fused + I) / 2

            result = torch.matmul(a, result)

    # CLS token attention ke semua patch
    mask = result[0, 1:]  # skip CLS token sendiri
    mask = mask / mask.max()
    return mask.numpy()


def visualize_attention_rollout(
    image: Image.Image,
    attention_mask: np.ndarray,
    patch_size: int = 16,
    title: str = "ViT Attention Rollout",
    figsize: tuple = (10, 5),
) -> plt.Figure:
    """Visualisasi attention rollout ViT pada gambar."""
    img_array = np.array(image.resize((224, 224)))
    grid_size = int(np.sqrt(len(attention_mask)))
    mask = attention_mask.reshape(grid_size, grid_size)

    from skimage.transform import resize as sk_resize
    mask_resized = sk_resize(mask, (224, 224))

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    axes[0].imshow(img_array)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(mask_resized, cmap="viridis")
    axes[1].set_title("Attention Map")
    axes[1].axis("off")

    axes[2].imshow(img_array)
    axes[2].imshow(mask_resized, cmap="jet", alpha=0.4)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig

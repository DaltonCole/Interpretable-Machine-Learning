"""
Chapter 28: Saliency Maps
https://christophm.github.io/interpretable-ml-book/pixel-wise.html

Saliency maps highlight which input pixels most influenced the model's prediction.
Methods: Vanilla Gradient, Integrated Gradients (via Captum).

Figures produced:
  1. saliency_grid.html/png      — vanilla gradients for several MNIST images
  2. integrated_grad.html/png    — integrated gradients for the same images
"""
import numpy as np
import plotly.graph_objects as go
import torch
import torch.nn.functional as F
from captum.attr import IntegratedGradients, Saliency
from plotly.subplots import make_subplots
from torch.utils.data import DataLoader, Subset

from ch27_learned_features.main import SimpleCNN, train_model
from shared.datasets import load_images
from shared.theme import save_figure

N_IMAGES = 6


def plot_saliency_grid(images: np.ndarray, saliency: np.ndarray,
                        labels: np.ndarray, preds: np.ndarray,
                        title: str) -> go.Figure:
    """Two-row grid: original images (top) and saliency maps (bottom)."""
    n = len(images)
    fig = make_subplots(
        rows=2, cols=n,
        row_titles=["Input", "Saliency"],
        subplot_titles=[f"True:{l} Pred:{p}" for l, p in zip(labels, preds)] + [""] * n,
    )
    for i in range(n):
        img = images[i].squeeze()
        sal = np.abs(saliency[i]).squeeze()
        sal = (sal - sal.min()) / (sal.max() - sal.min() + 1e-8)
        fig.add_trace(go.Heatmap(z=img, colorscale="Gray_r", showscale=False, hoverinfo="skip"),
                      row=1, col=i + 1)
        fig.add_trace(go.Heatmap(z=sal, colorscale="Hot", showscale=False, hoverinfo="skip"),
                      row=2, col=i + 1)
    fig.update_layout(title=title, height=380)
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False, autorange="reversed")
    return fig


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    train_ds, test_ds = load_images()
    print("Training SimpleCNN…")
    model = train_model(train_ds, epochs=3, device=device)
    model.eval()

    # Pick one image per class (digits 0-5)
    loader = DataLoader(test_ds, batch_size=1024)
    all_imgs, all_labels = [], []
    for X, y in loader:
        all_imgs.append(X); all_labels.append(y)
    all_imgs = torch.cat(all_imgs); all_labels = torch.cat(all_labels)

    selected = []
    for digit in range(N_IMAGES):
        idx = (all_labels == digit).nonzero(as_tuple=True)[0][0].item()
        selected.append(idx)
    X_sel = all_imgs[selected].to(device)
    y_sel = all_labels[selected]

    with torch.no_grad():
        preds_np = model(X_sel).argmax(dim=1).cpu().numpy()
    preds_list = preds_np.tolist()  # Captum requires Python ints, not numpy array

    print("Computing vanilla gradient saliency…")
    sal_method = Saliency(model)
    X_sel = X_sel.detach().requires_grad_(True)
    sal_attrs = sal_method.attribute(X_sel, target=preds_list).detach().cpu().numpy()

    print("Generating saliency grid…")
    fig1 = plot_saliency_grid(
        X_sel.detach().cpu().numpy(), sal_attrs, y_sel.numpy(), preds_np,
        title="Vanilla Gradient Saliency Maps",
    )
    save_figure(fig1, "ch28_saliency_maps", "saliency_grid")

    print("Computing integrated gradients…")
    ig = IntegratedGradients(model)
    baseline = torch.zeros_like(X_sel)
    ig_attrs = ig.attribute(X_sel, baseline, target=preds_list).detach().cpu().numpy()

    print("Generating integrated gradient grid…")
    fig2 = plot_saliency_grid(
        X_sel.detach().cpu().numpy(), ig_attrs, y_sel.numpy(), preds_np,
        title="Integrated Gradients",
    )
    save_figure(fig2, "ch28_saliency_maps", "integrated_grad")


if __name__ == "__main__":
    main()

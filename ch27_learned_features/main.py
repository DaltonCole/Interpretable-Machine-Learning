"""
Chapter 27: Learned Features
https://christophm.github.io/interpretable-ml-book/cnn-features.html

Visualises what a CNN has learned by showing:
  - Maximally activating dataset patches per filter
  - First-layer filter weights as images

Figures produced:
  1. filter_weights.html/png     — grid of first conv-layer filter weights
  2. max_activating.html/png     — top-k MNIST digits that maximise each filter
"""
import numpy as np
import plotly.graph_objects as go
import torch
import torch.nn as nn
import torch.nn.functional as F
from plotly.subplots import make_subplots
from torch.utils.data import DataLoader

from shared.datasets import load_images
from shared.theme import save_figure


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        return self.fc(x.flatten(1))


def train_model(train_ds, epochs: int = 3, device: str = "cpu"):
    loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    model = SimpleCNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(epochs):
        total_loss = 0
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            opt.zero_grad()
            loss = F.cross_entropy(model(X), y)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        print(f"  Epoch {epoch+1}/{epochs}  loss={total_loss/len(loader):.4f}")
    return model


def plot_filter_grid(weights: np.ndarray, n_filters: int = 16) -> go.Figure:
    """Display first conv-layer filter weights as a grid of grayscale images."""
    cols = 4
    rows = (n_filters + cols - 1) // cols
    fig = make_subplots(rows=rows, cols=cols,
                        subplot_titles=[f"Filter {i}" for i in range(n_filters)])
    for i in range(n_filters):
        w = weights[i, 0]  # (H, W)
        w = (w - w.min()) / (w.max() - w.min() + 1e-8)
        r, c = divmod(i, cols)
        fig.add_trace(go.Heatmap(
            z=w, colorscale="Gray", showscale=False,
            zmin=0, zmax=1, hoverinfo="skip",
        ), row=r + 1, col=c + 1)
    fig.update_layout(
        title="CNN First-Layer Learned Filters",
        height=rows * 180,
    )
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False, autorange="reversed")
    return fig


def plot_max_activating(images: np.ndarray, labels: np.ndarray,
                        filter_idx: int, n_show: int = 8) -> go.Figure:
    """Grid of digits that most strongly activate a given filter."""
    cols = n_show
    fig = make_subplots(rows=1, cols=cols,
                        subplot_titles=[f"Label: {labels[i]}" for i in range(n_show)])
    for i in range(n_show):
        img = images[i].squeeze()
        fig.add_trace(go.Heatmap(
            z=img, colorscale="Gray_r", showscale=False, hoverinfo="skip",
        ), row=1, col=i + 1)
    fig.update_layout(
        title=f"Top-{n_show} Images Maximising Filter {filter_idx}",
        height=200,
    )
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False, autorange="reversed")
    return fig


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    train_ds, test_ds = load_images()
    print("Training SimpleCNN on MNIST…")
    model = train_model(train_ds, epochs=3, device=device)
    model.eval()

    print("Generating filter weight grid…")
    weights = model.conv1.weight.detach().cpu().numpy()
    fig1 = plot_filter_grid(weights, n_filters=16)
    save_figure(fig1, "ch27_learned_features", "filter_weights")

    print("Finding maximally activating images for filter 0…")
    loader = DataLoader(test_ds, batch_size=512)
    all_acts, all_imgs, all_labels = [], [], []
    with torch.no_grad():
        for X, y in loader:
            acts = F.relu(model.conv1(X.to(device)))
            # Mean activation across spatial dims for each filter
            all_acts.append(acts[:, 0].mean(dim=(-2, -1)).cpu().numpy())
            all_imgs.append(X.numpy())
            all_labels.append(y.numpy())
            if len(all_imgs) * 512 > 5000:
                break

    all_acts = np.concatenate(all_acts)
    all_imgs = np.concatenate(all_imgs)
    all_labels = np.concatenate(all_labels)
    top_idx = np.argsort(all_acts)[-8:][::-1]

    fig2 = plot_max_activating(all_imgs[top_idx], all_labels[top_idx], filter_idx=0)
    save_figure(fig2, "ch27_learned_features", "max_activating")


if __name__ == "__main__":
    main()

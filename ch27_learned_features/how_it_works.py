"""
Chapter 27: How CNNs Learn Features
https://christophm.github.io/interpretable-ml-book/cnn-features.html

Conceptual figures explaining how a convolutional neural network builds up a
hierarchy of features from raw pixel patterns to class-level representations.

Figures produced:
  1. how_cnn_architecture.html/png  — architecture diagram (Plotly shapes + annotations)
  2. how_filter_effect.html/png     — original image | learned filter | feature map
  3. how_activation_space.html/png  — PCA of penultimate-layer activations by digit class
"""
import numpy as np
import plotly.graph_objects as go
import torch
import torch.nn as nn
import torch.nn.functional as F
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, Subset

from shared.datasets import load_images
from shared.theme import COLORS, save_figure

CHAPTER = "ch27_learned_features"


# ── SimpleCNN defined locally (ch27 is the canonical definition) ─────────────
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
        total_loss = 0.0
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            opt.zero_grad()
            loss = F.cross_entropy(model(X), y)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        print(f"  Epoch {epoch + 1}/{epochs}  loss={total_loss / len(loader):.4f}")
    return model


# ── Figure 1: Architecture Diagram ───────────────────────────────────────────

def plot_cnn_architecture() -> go.Figure:
    """Static architecture diagram with colored boxes and arrows."""
    stages = [
        dict(x=0.8,  label="Input<br>28×28",
             sub="Raw pixels",       color=COLORS["neutral"]),
        dict(x=2.4,  label="Conv1<br>16 filters 5×5<br>→ 28×28×16",
             sub="Detect edges",     color=COLORS["primary"]),
        dict(x=4.0,  label="MaxPool<br>→ 14×14×16",
             sub="Downsample",       color=COLORS["secondary"]),
        dict(x=5.6,  label="Conv2<br>32 filters 3×3<br>→ 14×14×32",
             sub="Detect textures",  color=COLORS["primary"]),
        dict(x=7.2,  label="MaxPool<br>→ 7×7×32",
             sub="Downsample",       color=COLORS["secondary"]),
        dict(x=8.8,  label="FC<br>1568 → 10",
             sub="Classify",         color=COLORS["accent"]),
        dict(x=10.4, label="Softmax<br>→ probabilities",
             sub="Class scores",     color=COLORS["positive"]),
    ]

    BOX_W = 1.2
    BOX_H = 0.80
    CY = 0.55

    fig = go.Figure()

    for s in stages:
        cx = s["x"]
        # Box
        fig.add_shape(
            type="rect",
            x0=cx - BOX_W / 2, x1=cx + BOX_W / 2,
            y0=CY - BOX_H / 2, y1=CY + BOX_H / 2,
            fillcolor=s["color"],
            line=dict(color="white", width=1.5),
            opacity=0.88,
        )
        # Main label inside box
        fig.add_annotation(
            x=cx, y=CY + 0.04,
            text=s["label"],
            showarrow=False,
            font=dict(size=10, color="white"),
            align="center",
        )
        # Sub-label below box
        fig.add_annotation(
            x=cx, y=CY - BOX_H / 2 - 0.13,
            text=f"<i>{s['sub']}</i>",
            showarrow=False,
            font=dict(size=10, color=COLORS["neutral"]),
            align="center",
        )

    # Arrows between consecutive boxes
    for i in range(len(stages) - 1):
        x0 = stages[i]["x"] + BOX_W / 2
        x1 = stages[i + 1]["x"] - BOX_W / 2
        fig.add_annotation(
            x=x1, y=CY,
            ax=x0, ay=CY,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=3, arrowwidth=2,
            arrowcolor=COLORS["neutral"],
            text="",
        )

    fig.update_layout(
        title=(
            "Convolutional Neural Network — Learned Feature Hierarchy<br>"
            "<sup>Each layer learns increasingly abstract representations: "
            "pixels → edges → textures → digit identity</sup>"
        ),
        xaxis=dict(visible=False, range=[0, 11.2]),
        yaxis=dict(visible=False, range=[0, 1.1]),
        height=300,
        margin=dict(l=20, r=20, t=90, b=70),
    )
    return fig


# ── Figure 2: Filter Effect ───────────────────────────────────────────────────

def plot_filter_effect(model, test_ds, device: str = "cpu") -> go.Figure:
    """3-panel: original image | learned conv1 filter weights | feature map."""
    targets = np.array(test_ds.targets)
    # Pick first correctly-classified digit 8
    img_tensor, label, pred = None, None, None
    model.eval()
    for idx in np.where(targets == 8)[0]:
        img_t, lbl = test_ds[idx]
        with torch.no_grad():
            p = model(img_t.unsqueeze(0).to(device)).argmax(1).item()
        if p == lbl:
            img_tensor, label, pred = img_t, int(lbl), p
            break

    img_np = img_tensor.squeeze().numpy()  # (28, 28)

    # Filter 0 weights from conv1 — shape (5, 5)
    filter_w = model.conv1.weight[0, 0].detach().cpu().numpy()

    # Feature map: conv1 → relu → maxpool, filter 0 → (14, 14)
    with torch.no_grad():
        x = img_tensor.unsqueeze(0).to(device)
        feat_map = F.relu(F.max_pool2d(model.conv1(x), 2))[0, 0].cpu().numpy()

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=[
            "Input Image",
            "Learned Filter (conv1, filter 0)",
            "Filter Response (feature map)",
        ],
        horizontal_spacing=0.08,
    )

    fig.add_trace(
        go.Heatmap(z=img_np, colorscale="Gray_r", showscale=False, hoverinfo="skip"),
        row=1, col=1,
    )
    fig.add_trace(
        go.Heatmap(z=filter_w, colorscale="RdBu", showscale=False, hoverinfo="skip"),
        row=1, col=2,
    )
    fig.add_trace(
        go.Heatmap(z=feat_map, colorscale="Hot", showscale=False, hoverinfo="skip"),
        row=1, col=3,
    )

    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False, autorange="reversed")

    fig.update_layout(
        title=(
            "Convolutional Filter — How a Filter Detects Local Patterns<br>"
            f"<sup>Digit '{label}': the 5×5 filter (center) responds strongly "
            "where its learned pattern matches strokes in the input (right)</sup>"
        ),
        height=380,
        margin=dict(t=100, b=50),
    )
    return fig


# ── Figure 3: Activation Space ────────────────────────────────────────────────

def plot_activation_space(model, test_ds, n_samples: int = 500,
                           device: str = "cpu") -> go.Figure:
    """PCA of penultimate-layer (flatten before fc) activations, one trace per digit."""
    activations: list[torch.Tensor] = []

    def _hook(module, inp, out):  # capture input to fc = flattened conv output
        activations.append(inp[0].detach().cpu())

    handle = model.fc.register_forward_hook(_hook)

    subset = Subset(test_ds, list(range(n_samples)))
    loader = DataLoader(subset, batch_size=128, shuffle=False)
    all_labels: list[int] = []
    model.eval()
    with torch.no_grad():
        for X, y in loader:
            model(X.to(device))
            all_labels.extend(y.numpy().tolist())
    handle.remove()

    acts = torch.cat(activations).numpy()   # (n_samples, 1568)
    labels = np.array(all_labels)

    pca = PCA(n_components=2, random_state=42)
    emb = pca.fit_transform(acts)
    var = pca.explained_variance_ratio_ * 100

    palette = COLORS["palette"]
    fig = go.Figure()
    for digit in range(10):
        mask = labels == digit
        fig.add_trace(go.Scatter(
            x=emb[mask, 0], y=emb[mask, 1],
            mode="markers",
            marker=dict(color=palette[digit % len(palette)], size=6, opacity=0.75),
            name=str(digit),
            hovertemplate=f"Digit {digit}<br>PC1=%{{x:.1f}}<br>PC2=%{{y:.1f}}<extra></extra>",
        ))

    fig.update_layout(
        title=(
            "Learned Features — The CNN's Internal Representation Separates Digit Classes<br>"
            f"<sup>PCA of 1568-dim penultimate activations | "
            f"PC1 {var[0]:.1f}% var · PC2 {var[1]:.1f}% var — "
            "distinct clusters show the network has learned meaningful features</sup>"
        ),
        xaxis_title=f"PC1 ({var[0]:.1f}% variance explained)",
        yaxis_title=f"PC2 ({var[1]:.1f}% variance explained)",
        legend=dict(orientation="h", y=-0.18, title="Digit"),
        height=520,
    )
    return fig


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    train_ds, test_ds = load_images()

    print("Figure 1: CNN architecture diagram…")
    fig1 = plot_cnn_architecture()
    save_figure(fig1, CHAPTER, "how_cnn_architecture")

    print("Training SimpleCNN (3 epochs)…")
    model = train_model(train_ds, epochs=3, device=device)
    model.eval()

    print("Figure 2: Filter effect…")
    fig2 = plot_filter_effect(model, test_ds, device=device)
    save_figure(fig2, CHAPTER, "how_filter_effect")

    print("Figure 3: Activation space (PCA)…")
    fig3 = plot_activation_space(model, test_ds, n_samples=500, device=device)
    save_figure(fig3, CHAPTER, "how_activation_space")


if __name__ == "__main__":
    main()

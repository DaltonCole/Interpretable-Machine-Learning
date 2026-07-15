"""
Chapter 29: How TCAV Works
https://christophm.github.io/interpretable-ml-book/tcav.html

Conceptual figures explaining concept activation vectors (CAVs) and the TCAV
method for testing how sensitive predictions are to user-defined concepts.

Figures produced:
  1. how_cav_concept.html/png    — PCA of conv2 activations: concept vs non-concept + CAV arrow
  2. how_tcav_score.html/png     — TCAV score heatmap (3 concepts × 3 target classes)
  3. how_tsne_concepts.html/png  — t-SNE of conv2 activations colored by digit class
"""
import numpy as np
import plotly.graph_objects as go
import torch
import torch.nn.functional as F
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, Subset

from ch27_learned_features.main import SimpleCNN, train_model
from shared.datasets import load_images
from shared.theme import COLORS, save_figure

CHAPTER = "ch29_concepts"


def extract_conv2_activations(model, dataset, device: str = "cpu"):
    """Flattened conv2 output activations and labels for every image in dataset.

    Returns (acts_flat: np.ndarray of shape (N, 6272), labels: np.ndarray).
    """
    captured: list[torch.Tensor] = []
    labels_out: list[int] = []

    def _hook(module, inp, out):
        captured.append(out.detach().cpu())

    handle = model.conv2.register_forward_hook(_hook)
    loader = DataLoader(dataset, batch_size=128, shuffle=False)
    model.eval()
    with torch.no_grad():
        for X, y in loader:
            model(X.to(device))
            labels_out.extend(y.numpy().tolist())
    handle.remove()

    acts = torch.cat(captured).numpy()          # (N, 32, 14, 14)
    acts_flat = acts.reshape(len(acts), -1)     # (N, 6272)
    return acts_flat, np.array(labels_out)


# ── Figure 1: CAV Direction ───────────────────────────────────────────────────

def plot_cav_concept(model, test_ds, device: str = "cpu") -> go.Figure:
    """PCA of conv2 activations: digit-0 concept (circles) vs non-concept (×)
    with the CAV direction drawn as an arrow."""
    targets = np.array(test_ds.targets)
    pos_idx = np.where(targets == 0)[0][:100].tolist()
    neg_idx = np.where(targets != 0)[0][:100].tolist()

    pos_ds = Subset(test_ds, pos_idx)
    neg_ds = Subset(test_ds, neg_idx)
    acts_pos, _ = extract_conv2_activations(model, pos_ds, device)
    acts_neg, _ = extract_conv2_activations(model, neg_ds, device)

    all_acts = np.vstack([acts_pos, acts_neg])
    y_cav = np.array([1] * len(acts_pos) + [0] * len(acts_neg))

    # Fit CAV
    clf = LogisticRegression(max_iter=500, random_state=42, C=1.0)
    clf.fit(all_acts, y_cav)
    cav = clf.coef_[0]
    cav /= np.linalg.norm(cav) + 1e-8

    # PCA for visualisation
    pca = PCA(n_components=2, random_state=42)
    emb = pca.fit_transform(all_acts)

    # Project the CAV direction into PCA space
    cav_2d = pca.transform(cav.reshape(1, -1))[0]
    center = emb.mean(axis=0)
    scale = 2.5 * np.std(emb, axis=0).mean()
    cav_end = center + scale * cav_2d / (np.linalg.norm(cav_2d) + 1e-8)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=emb[:len(acts_pos), 0], y=emb[:len(acts_pos), 1],
        mode="markers",
        marker=dict(symbol="circle", color=COLORS["primary"], size=7, opacity=0.7),
        name="Concept: digit 0",
        hovertemplate="Concept<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=emb[len(acts_pos):, 0], y=emb[len(acts_pos):, 1],
        mode="markers",
        marker=dict(symbol="x", color=COLORS["neutral"], size=7, opacity=0.6),
        name="Non-concept (other digits)",
        hovertemplate="Non-concept<extra></extra>",
    ))

    # CAV arrow
    fig.add_annotation(
        x=cav_end[0], y=cav_end[1],
        ax=center[0], ay=center[1],
        xref="x", yref="y", axref="x", ayref="y",
        arrowhead=3, arrowwidth=3,
        arrowcolor=COLORS["accent"],
        showarrow=True, text="",
    )
    fig.add_annotation(
        x=cav_end[0], y=cav_end[1],
        text="<b>CAV direction</b>",
        showarrow=False,
        font=dict(size=12, color=COLORS["accent"]),
        bgcolor="rgba(255,255,255,0.88)",
        bordercolor=COLORS["accent"], borderwidth=1,
        xanchor="left",
    )
    fig.add_annotation(
        x=0.5, y=-0.15, xref="paper", yref="paper",
        text=(
            "The CAV is the linear direction that best separates 'digit 0' from "
            "'other digits' in activation space"
        ),
        showarrow=False,
        font=dict(size=11, color=COLORS["neutral"]),
        bgcolor="rgba(255,255,255,0.85)",
        bordercolor="#D1D5DB", borderwidth=1,
        xanchor="center",
    )

    fig.update_layout(
        title=(
            "TCAV — A Concept Is a Direction in Activation Space<br>"
            "<sup>A logistic regression (CAV) learned on conv2 activations separates "
            "'digit 0' from random other images — the decision boundary is the concept</sup>"
        ),
        xaxis_title="PC1",
        yaxis_title="PC2",
        legend=dict(orientation="h", y=-0.22),
        height=520,
        margin=dict(b=90),
    )
    return fig


# ── Figure 2: TCAV Score Heatmap ──────────────────────────────────────────────

def _tcav_score(model, test_ds, concept_digit: int, target_class: int,
                device: str = "cpu", n_per_class: int = 100) -> float:
    """Simplified TCAV score: fraction of target-class activations aligned with CAV.

    TCAV = mean(acts_target · cav > 0) where cav is trained to separate the
    concept digit from random images.
    """
    targets = np.array(test_ds.targets)
    pos_idx = np.where(targets == concept_digit)[0][:n_per_class].tolist()
    neg_idx = np.where(targets != concept_digit)[0][:n_per_class].tolist()
    tgt_idx = np.where(targets == target_class)[0][:n_per_class].tolist()

    acts_pos, _ = extract_conv2_activations(model, Subset(test_ds, pos_idx), device)
    acts_neg, _ = extract_conv2_activations(model, Subset(test_ds, neg_idx), device)
    X_cav = np.vstack([acts_pos, acts_neg])
    y_cav = np.array([1] * len(acts_pos) + [0] * len(acts_neg))

    clf = LogisticRegression(max_iter=500, random_state=42, C=1.0)
    clf.fit(X_cav, y_cav)
    cav = clf.coef_[0]
    cav /= np.linalg.norm(cav) + 1e-8

    acts_tgt, _ = extract_conv2_activations(model, Subset(test_ds, tgt_idx), device)
    dots = acts_tgt @ cav
    return float((dots > 0).mean())


def plot_tcav_score(model, test_ds, device: str = "cpu") -> go.Figure:
    """Heatmap: rows = concept digits (0,1,2), cols = target classes (3,4,5)."""
    concepts = [0, 1, 2]
    targets_cls = [3, 4, 5]

    scores = np.zeros((len(concepts), len(targets_cls)))
    for i, cd in enumerate(concepts):
        for j, tc in enumerate(targets_cls):
            scores[i, j] = _tcav_score(model, test_ds, cd, tc, device)
            print(f"  concept={cd}  target={tc}  TCAV={scores[i,j]:.3f}")

    text = [[f"{scores[i, j]:.2f}" for j in range(len(targets_cls))]
            for i in range(len(concepts))]

    fig = go.Figure(go.Heatmap(
        z=scores,
        x=[f"Class {c}" for c in targets_cls],
        y=[f"Concept: digit {d}" for d in concepts],
        colorscale="RdBu",
        zmid=0.5, zmin=0.0, zmax=1.0,
        text=text,
        texttemplate="%{text}",
        colorbar=dict(title="TCAV<br>Score", thickness=16, len=0.75),
    ))

    fig.add_annotation(
        x=0.5, y=-0.22, xref="paper", yref="paper",
        text=(
            "Score > 0.5: concept direction is positively aligned with target-class activations  |  "
            "Score < 0.5: negatively aligned"
        ),
        showarrow=False,
        font=dict(size=11, color=COLORS["neutral"]),
        xanchor="center",
    )

    fig.update_layout(
        title=(
            "TCAV Score — Does the Concept Influence a Target Class's Predictions?<br>"
            "<sup>Score = fraction of target-class activations in the CAV direction</sup>"
        ),
        height=420,
        margin=dict(b=90, t=100),
    )
    return fig


# ── Figure 3: t-SNE Activation Space ──────────────────────────────────────────

def plot_tsne_concepts(model, test_ds, n_samples: int = 600,
                        device: str = "cpu") -> go.Figure:
    """t-SNE of conv2 activations for 600 test images, one trace per digit class."""
    subset = Subset(test_ds, list(range(n_samples)))
    acts_flat, labels = extract_conv2_activations(model, subset, device)

    # PCA pre-reduction for speed (standard practice before t-SNE)
    n_pca = min(50, acts_flat.shape[1])
    pca = PCA(n_components=n_pca, random_state=42)
    acts_pca = pca.fit_transform(acts_flat)

    print("  Running t-SNE (may take ~30 s)…")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=500)
    emb = tsne.fit_transform(acts_pca)

    palette = COLORS["palette"]
    fig = go.Figure()
    for digit in range(10):
        mask = labels == digit
        fig.add_trace(go.Scatter(
            x=emb[mask, 0], y=emb[mask, 1],
            mode="markers",
            marker=dict(color=palette[digit % len(palette)], size=4, opacity=0.80),
            name=str(digit),
            hovertemplate=f"Digit {digit}<extra></extra>",
        ))

    fig.add_annotation(
        x=0.5, y=0.02, xref="paper", yref="paper",
        text=(
            "Nearby points in t-SNE space → similar internal representations → similar 'concepts'"
        ),
        showarrow=False,
        font=dict(size=11, color=COLORS["neutral"]),
        bgcolor="rgba(255,255,255,0.85)",
        bordercolor="#D1D5DB", borderwidth=1,
        xanchor="center",
    )

    fig.update_layout(
        title=(
            "Activation Space — CNN Clusters Digits with Similar Concepts Together<br>"
            "<sup>t-SNE of conv2 activations (PCA-50 pre-reduced): each island "
            "corresponds to a distinct visual concept learned by the network</sup>"
        ),
        xaxis_title="t-SNE 1",
        yaxis_title="t-SNE 2",
        legend=dict(orientation="h", y=-0.18, title="Digit"),
        height=520,
    )
    return fig


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    train_ds, test_ds = load_images()
    print("Training SimpleCNN…")
    model = train_model(train_ds, epochs=3, device=device)
    model.eval()

    print("Figure 1: CAV concept…")
    fig1 = plot_cav_concept(model, test_ds, device=device)
    save_figure(fig1, CHAPTER, "how_cav_concept")

    print("Figure 2: TCAV score heatmap…")
    fig2 = plot_tcav_score(model, test_ds, device=device)
    save_figure(fig2, CHAPTER, "how_tcav_score")

    print("Figure 3: t-SNE activation space…")
    fig3 = plot_tsne_concepts(model, test_ds, n_samples=600, device=device)
    save_figure(fig3, CHAPTER, "how_tsne_concepts")


if __name__ == "__main__":
    main()

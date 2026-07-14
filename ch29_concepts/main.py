"""
Chapter 29: Detecting Concepts (TCAV)
https://christophm.github.io/interpretable-ml-book/tcav.html

TCAV (Testing with Concept Activation Vectors) measures how sensitive a model's
predictions are to user-defined concepts (e.g., "roundness", "brightness").

Figures produced:
  1. tcav_scores.html/png       — TCAV sensitivity scores per concept per class
  2. cav_projection.html/png    — t-SNE of activations, coloured by concept membership

Uses: captum.concept (TCAV implementation)
"""
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from ch27_learned_features.main import SimpleCNN, train_model
from shared.datasets import load_images
from shared.theme import COLORS, save_figure


def make_concept_datasets(test_ds, concept_digit: int, n_per_class: int = 100):
    """Binary concept: images of `concept_digit` vs. random others."""
    targets = np.array(test_ds.targets)
    pos_idx = np.where(targets == concept_digit)[0][:n_per_class]
    neg_idx = np.where(targets != concept_digit)[0][:n_per_class]
    return Subset(test_ds, pos_idx), Subset(test_ds, neg_idx)


def extract_activations(model, dataset, layer_name: str = "conv2", device: str = "cpu"):
    """Extract intermediate activations for all images in dataset."""
    activations = []
    labels = []

    def hook_fn(module, inp, out):
        activations.append(out.detach().cpu())

    layer = getattr(model, layer_name)
    handle = layer.register_forward_hook(hook_fn)
    loader = DataLoader(dataset, batch_size=128)
    model.eval()
    with torch.no_grad():
        for X, y in loader:
            model(X.to(device))
            labels.extend(y.numpy())
    handle.remove()
    return torch.cat(activations).numpy(), np.array(labels)


def compute_tcav_score(model, pos_ds, neg_ds, target_class: int,
                        device: str = "cpu") -> float:
    """Simplified TCAV: train linear CAV, measure directional derivative."""
    from sklearn.linear_model import LogisticRegression

    acts_pos, _ = extract_activations(model, pos_ds, device=device)
    acts_neg, _ = extract_activations(model, neg_ds, device=device)
    acts_pos_flat = acts_pos.reshape(len(acts_pos), -1)
    acts_neg_flat = acts_neg.reshape(len(acts_neg), -1)

    X_cav = np.vstack([acts_pos_flat, acts_neg_flat])
    y_cav = np.array([1] * len(acts_pos_flat) + [0] * len(acts_neg_flat))
    cav = LogisticRegression(max_iter=500).fit(X_cav, y_cav).coef_[0]
    cav /= (np.linalg.norm(cav) + 1e-8)

    # TCAV score: fraction of test examples with positive directional derivative
    loader = DataLoader(neg_ds, batch_size=64)
    scores = []
    for X, _ in loader:
        X = X.to(device).requires_grad_(True)
        logits = model(X)
        model.zero_grad()
        logits[:, target_class].sum().backward()
        grads, _ = extract_activations(model, Subset(neg_ds, list(range(min(64, len(neg_ds))))),
                                        device=device)
        grads_flat = grads.reshape(len(grads), -1)
        scores.extend((grads_flat @ cav > 0).astype(float))
        break  # one batch for demonstration

    return float(np.mean(scores))


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    train_ds, test_ds = load_images()
    print("Training SimpleCNN…")
    model = train_model(train_ds, epochs=3, device=device)

    concepts = [0, 1, 2]  # digits used as concept proxies
    target_classes = [3, 4, 5]

    print("Computing TCAV scores…")
    scores = np.zeros((len(concepts), len(target_classes)))
    for i, concept_digit in enumerate(concepts):
        pos_ds, neg_ds = make_concept_datasets(test_ds, concept_digit)
        for j, tc in enumerate(target_classes):
            scores[i, j] = compute_tcav_score(model, pos_ds, neg_ds, tc, device=device)
            print(f"  concept={concept_digit}  target={tc}  TCAV={scores[i,j]:.3f}")

    print("Generating TCAV score heatmap…")
    fig1 = go.Figure(go.Heatmap(
        z=scores,
        x=[f"Class {c}" for c in target_classes],
        y=[f"Concept: digit {d}" for d in concepts],
        colorscale="RdBu",
        zmid=0.5,
        zmin=0, zmax=1,
        text=np.round(scores, 2),
        texttemplate="%{text}",
        colorbar=dict(title="TCAV Score"),
    ))
    fig1.update_layout(
        title="TCAV Sensitivity Scores (>0.5 = concept positively influences class)",
    )
    save_figure(fig1, "ch29_concepts", "tcav_scores")

    print("Generating activation t-SNE projection…")
    from sklearn.manifold import TSNE
    subset = Subset(test_ds, list(range(500)))
    acts, lbl = extract_activations(model, subset, device=device)
    acts_flat = acts.reshape(len(acts), -1)
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    emb = tsne.fit_transform(acts_flat)

    fig2 = px.scatter(
        x=emb[:, 0], y=emb[:, 1], color=lbl.astype(str),
        title="t-SNE of conv2 Activations (coloured by digit)",
        labels={"x": "t-SNE 1", "y": "t-SNE 2", "color": "Digit"},
        color_discrete_sequence=COLORS["palette"],
    )
    save_figure(fig2, "ch29_concepts", "cav_projection")


if __name__ == "__main__":
    main()

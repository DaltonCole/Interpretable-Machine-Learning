"""
Chapter 26: Prototypes and Criticisms
https://christophm.github.io/interpretable-ml-book/proto.html

Prototypes are representative instances; criticisms are instances the prototypes
explain poorly. This chapter uses k-medoids (prototypes) and identifies outliers
(criticisms) by their distance to the nearest prototype.

Figures produced:
  1. prototypes_scatter.html/png — 2-D PCA projection with prototypes highlighted
  2. criticisms_bar.html/png     — most critical instances (farthest from prototypes)
"""
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler

from shared.datasets import load_regression
from shared.theme import COLORS, save_figure

N_PROTOTYPES = 5
N_CRITICISMS = 10


def kmedoids(X: np.ndarray, k: int, n_iter: int = 50, seed: int = 42):
    """Simple k-medoids via alternating assignment and medoid update."""
    rng = np.random.default_rng(seed)
    medoid_idx = rng.choice(len(X), k, replace=False)
    for _ in range(n_iter):
        D = pairwise_distances(X, X[medoid_idx])
        assignments = D.argmin(axis=1)
        new_medoids = []
        for j in range(k):
            cluster = np.where(assignments == j)[0]
            if len(cluster) == 0:
                new_medoids.append(medoid_idx[j])
                continue
            intra = pairwise_distances(X[cluster]).sum(axis=1)
            new_medoids.append(cluster[intra.argmin()])
        if np.array_equal(sorted(new_medoids), sorted(medoid_idx)):
            break
        medoid_idx = np.array(new_medoids)
    D = pairwise_distances(X, X[medoid_idx])
    assignments = D.argmin(axis=1)
    min_dists = D.min(axis=1)
    return medoid_idx, assignments, min_dists


def main():
    X_train, X_test, y_train, y_test, feature_names = load_regression()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    print(f"Finding {N_PROTOTYPES} prototypes via k-medoids…")
    proto_idx, assignments, min_dists = kmedoids(X_scaled, N_PROTOTYPES)
    print(f"  Prototypes at training indices: {proto_idx.tolist()}")

    criticism_idx = np.argsort(min_dists)[-N_CRITICISMS:][::-1]

    # 2-D PCA projection
    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X_scaled)

    print("Generating prototype scatter…")
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=X_2d[:, 0], y=X_2d[:, 1], mode="markers",
        marker=dict(color=assignments, colorscale="Viridis", size=4, opacity=0.4),
        name="Instances",
        hovertemplate="y=%{text}<extra></extra>",
        text=[f"{v:.0f}" for v in y_train],
    ))
    fig1.add_trace(go.Scatter(
        x=X_2d[proto_idx, 0], y=X_2d[proto_idx, 1], mode="markers+text",
        text=[f"P{i}" for i in range(N_PROTOTYPES)],
        textposition="top center",
        marker=dict(color=COLORS["accent"], size=16, symbol="star", line=dict(width=2, color="white")),
        name="Prototypes",
    ))
    fig1.add_trace(go.Scatter(
        x=X_2d[criticism_idx, 0], y=X_2d[criticism_idx, 1], mode="markers",
        marker=dict(color=COLORS["negative"], size=10, symbol="x", line=dict(width=2)),
        name="Criticisms",
    ))
    fig1.update_layout(
        title=f"Prototypes (★) and Criticisms (✗) — PCA Projection",
        xaxis_title=f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var.)",
        yaxis_title=f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var.)",
    )
    save_figure(fig1, "ch26_prototypes", "prototypes_scatter")

    print("Generating criticisms distance bar…")
    fig2 = go.Figure(go.Bar(
        x=min_dists[criticism_idx],
        y=[f"#{i}  (cnt={y_train[i]:.0f})" for i in criticism_idx],
        orientation="h",
        marker_color=COLORS["negative"],
        hovertemplate="Instance %{y}<br>Distance: %{x:.3f}<extra></extra>",
    ))
    fig2.update_layout(
        title=f"Top {N_CRITICISMS} Criticisms — Distance to Nearest Prototype",
        xaxis_title="Distance to nearest prototype (standardised space)",
    )
    save_figure(fig2, "ch26_prototypes", "criticisms_bar")


if __name__ == "__main__":
    main()

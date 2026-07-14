"""
Chapter 21: Feature Interaction
https://christophm.github.io/interpretable-ml-book/interaction.html

The H-statistic measures how much of a model's prediction variance is due
to interactions between features (vs. individual main effects).

Figures produced:
  1. h_stat_pairwise.html/png — pairwise H-statistics heatmap
  2. h_stat_total.html/png    — total interaction strength per feature (bar chart)
"""
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from shared.datasets import load_regression
from shared.models import get_regression_model
from shared.theme import COLORS, save_figure

N_SAMPLE = 300  # subsample for speed
N_GRID = 20


def pdp_1d(model, X, fi, grid):
    out = []
    for v in grid:
        Xc = X.copy()
        Xc[:, fi] = v
        out.append(model.predict(Xc).mean())
    return np.array(out)


def pdp_2d(model, X, fi, fj, gi, gj):
    Z = np.zeros((len(gi), len(gj)))
    for r, vi in enumerate(gi):
        Xc = X.copy()
        Xc[:, fi] = vi
        for c, vj in enumerate(gj):
            Xc2 = Xc.copy()
            Xc2[:, fj] = vj
            Z[r, c] = model.predict(Xc2).mean()
    return Z


def h_stat_pair(model, X, fi, fj):
    """Friedman H-statistic for the (fi, fj) pair. Values in [0, 1]."""
    gi = np.linspace(X[:, fi].min(), X[:, fi].max(), N_GRID)
    gj = np.linspace(X[:, fj].min(), X[:, fj].max(), N_GRID)
    pd_ij = pdp_2d(model, X, fi, fj, gi, gj)
    pd_i = pdp_1d(model, X, fi, gi)
    pd_j = pdp_1d(model, X, fj, gj)
    # Numerator: (pd_ij - pd_i[:, None] - pd_j[None, :])^2
    diff = pd_ij - pd_i[:, None] - pd_j[None, :]
    numerator = (diff ** 2).mean()
    denominator = (pd_ij ** 2).mean()
    return float(numerator / (denominator + 1e-10))


def main():
    X_train, X_test, y_train, y_test, feature_names = load_regression()
    model = get_regression_model(X_train, y_train)

    rng = np.random.default_rng(42)
    idx = rng.choice(len(X_train), N_SAMPLE, replace=False)
    X_sub = X_train[idx]

    # Use top features by importance to keep computation tractable
    importances = model.feature_importances_
    top_n = 6
    top_indices = list(np.argsort(importances)[-top_n:])
    top_names = [feature_names[i] for i in top_indices]

    print(f"Computing pairwise H-statistics for {top_n} features ({top_n*(top_n-1)//2} pairs)…")
    H = np.zeros((top_n, top_n))
    for a in range(top_n):
        for b in range(a + 1, top_n):
            h = h_stat_pair(model, X_sub, top_indices[a], top_indices[b])
            H[a, b] = H[b, a] = h
            print(f"  {top_names[a]} × {top_names[b]}: {h:.3f}")

    print("Generating pairwise H-stat heatmap…")
    fig1 = go.Figure(go.Heatmap(
        z=H,
        x=top_names,
        y=top_names,
        colorscale="Blues",
        zmin=0,
        text=np.round(H, 3),
        texttemplate="%{text}",
        colorbar=dict(title="H-statistic"),
        hovertemplate="%{y} × %{x}: %{z:.3f}<extra></extra>",
    ))
    fig1.update_layout(title="Pairwise Feature Interaction — H-Statistic")
    save_figure(fig1, "ch21_feature_interaction", "h_stat_pairwise")

    print("Generating total interaction bar chart…")
    total_h = H.sum(axis=1)
    order = np.argsort(total_h)
    fig2 = go.Figure(go.Bar(
        x=total_h[order],
        y=[top_names[i] for i in order],
        orientation="h",
        marker_color=COLORS["primary"],
    ))
    fig2.update_layout(
        title="Total Interaction Strength per Feature (sum of pairwise H)",
        xaxis_title="Total H-statistic",
    )
    save_figure(fig2, "ch21_feature_interaction", "h_stat_total")


if __name__ == "__main__":
    main()

"""
Chapter 19: Partial Dependence Plot (PDP)
https://christophm.github.io/interpretable-ml-book/pdp.html

A PDP shows the marginal effect of one (or two) features on the predicted outcome
by averaging over all other features in the training data.

Figures produced:
  1. pdp_1d.html/png   — 1-D PDP for the top-4 most important features (subplots)
  2. pdp_2d.html/png   — 2-D PDP heatmap for the two most important features
"""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from shared.datasets import load_regression
from shared.models import get_regression_model
from shared.theme import COLORS, save_figure

N_GRID = 50   # grid resolution per feature


def pdp_1d(model, X: np.ndarray, feature_idx: int, n_grid: int = N_GRID):
    """Marginal effect of one feature (Monte Carlo average over X)."""
    grid = np.linspace(X[:, feature_idx].min(), X[:, feature_idx].max(), n_grid)
    avg_preds = []
    for val in grid:
        X_mod = X.copy()
        X_mod[:, feature_idx] = val
        avg_preds.append(model.predict(X_mod).mean())
    return grid, np.array(avg_preds)


def pdp_2d(model, X: np.ndarray, fi: int, fj: int, n_grid: int = 30):
    """Joint marginal effect of two features."""
    gi = np.linspace(X[:, fi].min(), X[:, fi].max(), n_grid)
    gj = np.linspace(X[:, fj].min(), X[:, fj].max(), n_grid)
    Z = np.zeros((n_grid, n_grid))
    for r, vi in enumerate(gi):
        X_mod = X.copy()
        X_mod[:, fi] = vi
        for c, vj in enumerate(gj):
            X_mod2 = X_mod.copy()
            X_mod2[:, fj] = vj
            Z[r, c] = model.predict(X_mod2).mean()
    return gi, gj, Z


def plot_pdp_1d(model, X, feature_names, top_indices) -> go.Figure:
    cols = 2
    rows = (len(top_indices) + 1) // cols
    fig = make_subplots(rows=rows, cols=cols,
                        subplot_titles=[feature_names[i] for i in top_indices],
                        vertical_spacing=0.14)
    for k, fi in enumerate(top_indices):
        grid, avg = pdp_1d(model, X, fi)
        r, c = divmod(k, cols)
        fig.add_trace(go.Scatter(x=grid, y=avg, mode="lines",
                                  line=dict(color=COLORS["primary"], width=3),
                                  showlegend=False),
                      row=r + 1, col=c + 1)
        # Rug: actual data distribution
        fig.add_trace(go.Scatter(
            x=X[:, fi], y=[avg.min() - (avg.max() - avg.min()) * 0.05] * len(X),
            mode="markers",
            marker=dict(color=COLORS["neutral"], size=3, opacity=0.3, symbol="line-ns-open"),
            showlegend=False, hoverinfo="skip",
        ), row=r + 1, col=c + 1)
    fig.update_layout(title="Partial Dependence Plots — Top Features", height=rows * 280)
    return fig


def plot_pdp_2d(model, X, feature_names, fi, fj) -> go.Figure:
    gi, gj, Z = pdp_2d(model, X, fi, fj)
    fig = go.Figure(go.Heatmap(
        x=gj, y=gi, z=Z,
        colorscale="RdBu_r",
        colorbar=dict(title="Predicted count"),
        hovertemplate=(
            f"{feature_names[fi]}=%{{y:.2f}}<br>"
            f"{feature_names[fj]}=%{{x:.2f}}<br>"
            "Prediction=%{z:.1f}<extra></extra>"
        ),
    ))
    fig.update_layout(
        title=f"2-D PDP — {feature_names[fi]} × {feature_names[fj]}",
        xaxis_title=feature_names[fj],
        yaxis_title=feature_names[fi],
    )
    return fig


def main():
    X_train, X_test, y_train, y_test, feature_names = load_regression()
    model = get_regression_model(X_train, y_train)

    importances = model.feature_importances_
    top4 = list(np.argsort(importances)[-4:])
    top2 = top4[-2:]

    print("Computing 1-D PDPs…")
    fig1 = plot_pdp_1d(model, X_train, feature_names, top4)
    save_figure(fig1, "ch19_pdp", "pdp_1d")

    print("Computing 2-D PDP…")
    fig2 = plot_pdp_2d(model, X_train, feature_names, top2[0], top2[1])
    save_figure(fig2, "ch19_pdp", "pdp_2d")


if __name__ == "__main__":
    main()

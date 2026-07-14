"""
Chapter 22: Functional Decomposition
https://christophm.github.io/interpretable-ml-book/decomposition.html

Functional decomposition breaks a model's prediction function into main effects
(one feature at a time) and interaction effects. This chapter visualises main
effects alongside the residual (captured by interactions).

Figures produced:
  1. main_effects.html/png     — main effect function per feature (subplots)
  2. decomposition_r2.html/png — variance explained by each component
"""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from shared.datasets import load_regression
from shared.models import get_regression_model
from shared.theme import COLORS, save_figure

N_GRID = 60
N_SAMPLE = 500


def main_effect(model, X: np.ndarray, feature_idx: int, n_grid: int = N_GRID):
    """Main effect: centred PDP (removes the intercept term)."""
    grid = np.linspace(X[:, feature_idx].min(), X[:, feature_idx].max(), n_grid)
    preds = []
    for v in grid:
        Xc = X.copy(); Xc[:, feature_idx] = v
        preds.append(model.predict(Xc).mean())
    arr = np.array(preds)
    return grid, arr - arr.mean()


def plot_main_effects(model, X, feature_names, top_indices) -> go.Figure:
    cols = 3
    rows = (len(top_indices) + cols - 1) // cols
    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=[feature_names[i] for i in top_indices],
        vertical_spacing=0.12,
    )
    for k, fi in enumerate(top_indices):
        grid, effect = main_effect(model, X, fi)
        r, c = divmod(k, cols)
        fig.add_trace(go.Scatter(
            x=grid, y=effect, mode="lines",
            line=dict(color=COLORS["primary"], width=2), showlegend=False,
        ), row=r + 1, col=c + 1)
        fig.add_hline(y=0, line_dash="dot", line_color=COLORS["neutral"],
                      row=r + 1, col=c + 1)
    fig.update_layout(title="Functional Decomposition — Main Effects", height=rows * 240)
    return fig


def plot_decomposition_r2(model, X, y, feature_names, top_indices) -> go.Figure:
    """Bar chart of variance explained by each feature's main effect."""
    f_hat = model.predict(X)
    total_var = np.var(f_hat)
    r2s = []
    for fi in top_indices:
        _, effect_grid = main_effect(model, X, fi)
        # Assign main effect to each instance by nearest grid point
        grid = np.linspace(X[:, fi].min(), X[:, fi].max(), N_GRID)
        assignments = np.searchsorted(grid, X[:, fi]).clip(0, N_GRID - 1)
        instance_effects = effect_grid[assignments]
        r2s.append(np.var(instance_effects) / (total_var + 1e-10))

    order = np.argsort(r2s)
    fig = go.Figure(go.Bar(
        x=[r2s[i] for i in order],
        y=[feature_names[top_indices[i]] for i in order],
        orientation="h",
        marker_color=COLORS["primary"],
        text=[f"{r2s[i]:.1%}" for i in order],
        textposition="outside",
    ))
    fig.update_layout(
        title="Variance Explained by Each Main Effect",
        xaxis_title="Fraction of prediction variance",
        xaxis_tickformat=".0%",
    )
    return fig


def main():
    X_train, X_test, y_train, y_test, feature_names = load_regression()
    model = get_regression_model(X_train, y_train)

    rng = np.random.default_rng(42)
    idx = rng.choice(len(X_train), N_SAMPLE, replace=False)
    X_sub = X_train[idx]

    importances = model.feature_importances_
    top_indices = list(np.argsort(importances)[-9:])  # top 9

    print("Computing main effects…")
    fig1 = plot_main_effects(model, X_sub, feature_names, top_indices)
    save_figure(fig1, "ch22_functional_decomposition", "main_effects")

    print("Computing variance decomposition…")
    fig2 = plot_decomposition_r2(model, X_sub, y_train[idx], feature_names, top_indices)
    save_figure(fig2, "ch22_functional_decomposition", "decomposition_r2")


if __name__ == "__main__":
    main()

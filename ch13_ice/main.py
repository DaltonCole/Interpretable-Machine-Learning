"""
Chapter 13: Individual Conditional Expectation (ICE)
https://christophm.github.io/interpretable-ml-book/ice.html

ICE plots show one prediction curve per instance as a single feature varies.
Centred ICE (c-ICE) removes the vertical offset to highlight shape differences.

Figures produced:
  1. ice_plot.html/png    — ICE + PDP overlay for the most important feature
  2. cice_plot.html/png   — Centred ICE for the same feature
"""
import numpy as np
import plotly.graph_objects as go

from shared.datasets import load_regression
from shared.models import get_regression_model
from shared.theme import COLORS, save_figure

N_INSTANCES = 200  # lines to draw (random sample for readability)
N_POINTS = 80      # grid points along the feature axis


def compute_ice(model, X: np.ndarray, feature_idx: int, n_points: int = N_POINTS):
    """Return (grid_values, ice_matrix) where ice_matrix is shape (n_instances, n_points)."""
    grid = np.linspace(X[:, feature_idx].min(), X[:, feature_idx].max(), n_points)
    ice = np.zeros((len(X), n_points))
    for j, val in enumerate(grid):
        X_mod = X.copy()
        X_mod[:, feature_idx] = val
        ice[:, j] = model.predict(X_mod)
    return grid, ice


def plot_ice(grid, ice, pdp, feature_name: str, y_label: str = "Predicted count") -> go.Figure:
    fig = go.Figure()
    for i in range(len(ice)):
        fig.add_trace(go.Scatter(
            x=grid, y=ice[i],
            mode="lines",
            line=dict(color="rgba(37,99,235,0.15)", width=1),
            showlegend=False,
            hoverinfo="skip",
        ))
    fig.add_trace(go.Scatter(
        x=grid, y=pdp, mode="lines",
        line=dict(color=COLORS["accent"], width=3),
        name="PDP (average)",
    ))
    fig.update_layout(
        title=f"ICE Plot — {feature_name}",
        xaxis_title=feature_name,
        yaxis_title=y_label,
    )
    return fig


def plot_cice(grid, ice, feature_name: str) -> go.Figure:
    """Centre each curve at the leftmost grid point."""
    cice = ice - ice[:, [0]]
    pdp = cice.mean(axis=0)
    fig = go.Figure()
    for i in range(len(cice)):
        fig.add_trace(go.Scatter(
            x=grid, y=cice[i],
            mode="lines",
            line=dict(color="rgba(37,99,235,0.15)", width=1),
            showlegend=False,
            hoverinfo="skip",
        ))
    fig.add_trace(go.Scatter(
        x=grid, y=pdp, mode="lines",
        line=dict(color=COLORS["accent"], width=3),
        name="c-PDP (centred average)",
    ))
    fig.add_hline(y=0, line_dash="dot", line_color=COLORS["neutral"])
    fig.update_layout(
        title=f"Centred ICE Plot — {feature_name}",
        xaxis_title=feature_name,
        yaxis_title="Δ prediction from leftmost value",
    )
    return fig


def main():
    X_train, X_test, y_train, y_test, feature_names = load_regression()
    model = get_regression_model(X_train, y_train)

    # Use the most important feature
    top_idx = int(np.argmax(model.feature_importances_))
    feat_name = feature_names[top_idx]
    print(f"Most important feature: {feat_name}")

    rng = np.random.default_rng(42)
    sample_idx = rng.choice(len(X_test), size=min(N_INSTANCES, len(X_test)), replace=False)
    X_sample = X_test[sample_idx]

    print("Computing ICE curves…")
    grid, ice = compute_ice(model, X_sample, top_idx)
    pdp = ice.mean(axis=0)

    print("Generating ICE plot…")
    fig1 = plot_ice(grid, ice, pdp, feat_name)
    save_figure(fig1, "ch13_ice", "ice_plot")

    print("Generating centred ICE plot…")
    fig2 = plot_cice(grid, ice, feat_name)
    save_figure(fig2, "ch13_ice", "cice_plot")


if __name__ == "__main__":
    main()

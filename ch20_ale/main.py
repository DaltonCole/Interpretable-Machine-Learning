"""
Chapter 20: Accumulated Local Effects (ALE)
https://christophm.github.io/interpretable-ml-book/ale.html

ALE plots are like PDPs but avoid the problem of averaging over unrealistic
feature combinations. They compute local differences within intervals and
accumulate them — producing an unbiased marginal effect estimate.

Figures produced:
  1. ale_1d.html/png      — ALE curves for top features (subplots)
  2. ale_vs_pdp.html/png  — ALE vs. PDP overlay for the most important feature
"""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from shared.datasets import load_regression
from shared.models import get_regression_model
from shared.theme import COLORS, save_figure

N_INTERVALS = 40


def ale_1d(model, X: np.ndarray, feature_idx: int, n_intervals: int = N_INTERVALS):
    """Compute ALE for one feature. Returns (bin_centres, ale_values)."""
    quantiles = np.percentile(X[:, feature_idx], np.linspace(0, 100, n_intervals + 1))
    quantiles = np.unique(quantiles)
    centres, effects = [], []
    for lo, hi in zip(quantiles[:-1], quantiles[1:]):
        mask = (X[:, feature_idx] >= lo) & (X[:, feature_idx] < hi)
        if mask.sum() == 0:
            continue
        X_lo = X[mask].copy()
        X_hi = X[mask].copy()
        X_lo[:, feature_idx] = lo
        X_hi[:, feature_idx] = hi
        effects.append((model.predict(X_hi) - model.predict(X_lo)).mean())
        centres.append((lo + hi) / 2)

    ale = np.cumsum(effects)
    ale -= ale.mean()  # centre at zero
    return np.array(centres), ale


def pdp_1d(model, X: np.ndarray, feature_idx: int, grid: np.ndarray):
    """PDP on the same grid as ALE for direct comparison."""
    avg_preds = []
    for val in grid:
        X_mod = X.copy()
        X_mod[:, feature_idx] = val
        avg_preds.append(model.predict(X_mod).mean())
    avg = np.array(avg_preds)
    return avg - avg.mean()  # centre for fair comparison


def plot_ale_subplots(model, X, feature_names, top_indices) -> go.Figure:
    cols = 2
    rows = (len(top_indices) + 1) // cols
    fig = make_subplots(rows=rows, cols=cols,
                        subplot_titles=[feature_names[i] for i in top_indices],
                        vertical_spacing=0.14)
    for k, fi in enumerate(top_indices):
        centres, ale = ale_1d(model, X, fi)
        r, c = divmod(k, cols)
        fig.add_trace(go.Scatter(x=centres, y=ale, mode="lines",
                                  line=dict(color=COLORS["primary"], width=3),
                                  showlegend=False),
                      row=r + 1, col=c + 1)
        fig.add_hline(y=0, line_dash="dot", line_color=COLORS["neutral"],
                      row=r + 1, col=c + 1)
    fig.update_layout(title="ALE Plots — Top Features", height=rows * 280)
    return fig


def plot_ale_vs_pdp(model, X, feature_names, fi) -> go.Figure:
    centres, ale = ale_1d(model, X, fi)
    pdp = pdp_1d(model, X, fi, centres)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=centres, y=ale, mode="lines",
                              line=dict(color=COLORS["primary"], width=3), name="ALE"))
    fig.add_trace(go.Scatter(x=centres, y=pdp, mode="lines",
                              line=dict(color=COLORS["accent"], width=3, dash="dash"), name="PDP"))
    fig.add_hline(y=0, line_dash="dot", line_color=COLORS["neutral"])
    fig.update_layout(
        title=f"ALE vs. PDP — {feature_names[fi]}",
        xaxis_title=feature_names[fi],
        yaxis_title="Centred effect on predicted count",
    )
    return fig


def main():
    X_train, X_test, y_train, y_test, feature_names = load_regression()
    model = get_regression_model(X_train, y_train)

    importances = model.feature_importances_
    top4 = list(np.argsort(importances)[-4:])
    top1 = top4[-1]

    print("Computing ALE subplots…")
    fig1 = plot_ale_subplots(model, X_train, feature_names, top4)
    save_figure(fig1, "ch20_ale", "ale_1d")

    print("Computing ALE vs. PDP comparison…")
    fig2 = plot_ale_vs_pdp(model, X_train, feature_names, top1)
    save_figure(fig2, "ch20_ale", "ale_vs_pdp")


if __name__ == "__main__":
    main()

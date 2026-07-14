"""
Chapter 12: Ceteris Paribus Profiles
https://christophm.github.io/interpretable-ml-book/ceteris-paribus.html

A Ceteris Paribus (CP) profile shows how the prediction for a single instance
changes as one feature is varied while all others are held fixed.

Figures produced:
  1. cp_profiles.html/png — CP profiles for a few features for one instance
  2. cp_multifeature.html/png — CP profile grid across all features
"""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from shared.datasets import load_regression
from shared.models import get_regression_model
from shared.theme import COLORS, save_figure


def compute_cp_profile(model, instance: np.ndarray, X_train: np.ndarray,
                        feature_idx: int, n_points: int = 100):
    """Vary feature_idx across its training range; return (values, predictions)."""
    feat_min, feat_max = X_train[:, feature_idx].min(), X_train[:, feature_idx].max()
    values = np.linspace(feat_min, feat_max, n_points)

    X_sweep = np.tile(instance, (n_points, 1))
    X_sweep[:, feature_idx] = values

    return values, model.predict(X_sweep)


def plot_cp_profiles(model, instance, X_train, feature_names, feature_indices) -> go.Figure:
    """Overlay CP profiles for several features on one plot.

    X is normalised to [0, 1] per feature so profiles with different raw scales
    can be compared directly on a shared axis.
    """
    fig = go.Figure()
    original_pred = model.predict(instance.reshape(1, -1))[0]

    for idx in feature_indices:
        values, preds = compute_cp_profile(model, instance, X_train, idx)
        v_min, v_max = values.min(), values.max()
        x_norm = (values - v_min) / (v_max - v_min + 1e-8)
        orig_norm = (instance[idx] - v_min) / (v_max - v_min + 1e-8)
        fig.add_trace(go.Scatter(
            x=x_norm, y=preds,
            mode="lines", name=feature_names[idx], line=dict(width=2),
        ))
        fig.add_trace(go.Scatter(
            x=[orig_norm], y=[original_pred],
            mode="markers", showlegend=False,
            marker=dict(size=10, symbol="x", color=COLORS["neutral"]),
        ))

    fig.add_hline(y=original_pred, line_dash="dot", line_color=COLORS["neutral"],
                  annotation_text="original prediction")
    fig.update_layout(
        title="Ceteris Paribus Profiles — top features (x normalised to [0 = min, 1 = max])",
        xaxis_title="Normalised feature value",
        yaxis_title="Predicted rental count",
    )
    return fig


def plot_cp_grid(model, instance, X_train, feature_names) -> go.Figure:
    """One CP profile subplot per feature."""
    n = len(feature_names)
    cols = 3
    rows = (n + cols - 1) // cols
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=feature_names,
                        vertical_spacing=0.1, horizontal_spacing=0.08)
    original_pred = model.predict(instance.reshape(1, -1))[0]

    for i, name in enumerate(feature_names):
        values, preds = compute_cp_profile(model, instance, X_train, i)
        r, c = divmod(i, cols)
        fig.add_trace(go.Scatter(x=values, y=preds, mode="lines",
                                  line=dict(color=COLORS["primary"], width=2),
                                  showlegend=False),
                      row=r + 1, col=c + 1)
        fig.add_trace(go.Scatter(x=[instance[i]], y=[original_pred],
                                  mode="markers", showlegend=False,
                                  marker=dict(size=8, color=COLORS["accent"])),
                      row=r + 1, col=c + 1)

    fig.update_layout(title="Ceteris Paribus — All Features", height=rows * 220)
    return fig


def main():
    X_train, X_test, y_train, y_test, feature_names = load_regression()
    model = get_regression_model(X_train, y_train)

    instance = X_test[0]
    print(f"Instance prediction: {model.predict(instance.reshape(1, -1))[0]:.1f}")

    # Top 4 features by importance for the overlay plot
    importances = model.feature_importances_
    top_indices = np.argsort(importances)[-4:]

    print("Generating CP profile overlay…")
    fig1 = plot_cp_profiles(model, instance, X_train, feature_names, top_indices)
    save_figure(fig1, "ch12_ceteris_paribus", "cp_profiles")

    print("Generating CP grid for all features…")
    fig2 = plot_cp_grid(model, instance, X_train, feature_names)
    save_figure(fig2, "ch12_ceteris_paribus", "cp_multifeature")


if __name__ == "__main__":
    main()

"""
Chapter 17: Shapley Values
https://christophm.github.io/interpretable-ml-book/shapley.html

Shapley values are the game-theoretic optimal way to distribute a model's
prediction among its input features. This chapter computes them exactly
via all coalitions (feasible only for small feature sets).

Figures produced:
  1. shapley_waterfall.html/png   — waterfall (contribution) chart for one instance
  2. shapley_all_features.html/png — Shapley values across all test instances (violin)
"""
import itertools
import math
from functools import lru_cache

import numpy as np
import plotly.graph_objects as go

from shared.datasets import load_regression
from shared.models import get_regression_model
from shared.theme import COLORS, save_figure

# Use a small feature subset for exact Shapley computation (2^n coalitions)
N_FEATURES_EXACT = 8


def shapley_exact(model, instance: np.ndarray, X_train: np.ndarray,
                  feature_indices: list) -> np.ndarray:
    """Exact Shapley values for `feature_indices` features via all 2^n coalitions."""
    n = len(feature_indices)
    phi = np.zeros(n)
    baseline = X_train.mean(axis=0)

    for i, fi in enumerate(feature_indices):
        others = [fj for fj in feature_indices if fj != fi]
        for size in range(len(others) + 1):
            for coalition in itertools.combinations(others, size):
                weight = (math.factorial(size) * math.factorial(n - size - 1)
                          / math.factorial(n))
                # With feature i
                x_with = baseline.copy()
                for fj in coalition:
                    x_with[fj] = instance[fj]
                x_with[fi] = instance[fi]
                # Without feature i
                x_without = baseline.copy()
                for fj in coalition:
                    x_without[fj] = instance[fj]

                phi[i] += weight * (model.predict(x_with.reshape(1, -1))[0]
                                    - model.predict(x_without.reshape(1, -1))[0])
    return phi


def plot_waterfall(phi: np.ndarray, feature_names: list,
                   baseline: float, prediction: float) -> go.Figure:
    """Waterfall chart showing how each feature contribution builds from baseline."""
    order = np.argsort(np.abs(phi))[::-1]
    phi_sorted = phi[order]
    names_sorted = [feature_names[i] for i in order]

    running = baseline
    fig = go.Figure()
    fig.add_trace(go.Bar(x=["Baseline"], y=[baseline], marker_color=COLORS["neutral"], name="Baseline"))

    for i, (name, val) in enumerate(zip(names_sorted, phi_sorted)):
        color = COLORS["positive"] if val >= 0 else COLORS["negative"]
        fig.add_trace(go.Bar(
            x=[name], y=[val], base=[running],
            marker_color=color, name=name,
            hovertemplate=f"{name}: {val:+.2f}<extra></extra>",
        ))
        running += val

    fig.add_trace(go.Bar(x=["Prediction"], y=[prediction], marker_color=COLORS["primary"], name="Prediction"))
    fig.update_layout(
        title=f"Shapley Values — Waterfall  (baseline={baseline:.1f}, pred={prediction:.1f})",
        yaxis_title="Predicted rental count",
        barmode="stack",
        showlegend=False,
    )
    return fig


def main():
    X_train, X_test, y_train, y_test, feature_names = load_regression()
    model = get_regression_model(X_train, y_train)

    # Select top-N most important features for exact computation
    importances = model.feature_importances_
    top_indices = list(np.argsort(importances)[-N_FEATURES_EXACT:])
    top_names = [feature_names[i] for i in top_indices]

    instance = X_test[0]
    prediction = model.predict(instance.reshape(1, -1))[0]
    baseline_pred = model.predict(X_train.mean(axis=0).reshape(1, -1))[0]

    print(f"Computing exact Shapley values for {N_FEATURES_EXACT} features…")
    phi = shapley_exact(model, instance, X_train, top_indices)
    print(f"  Sum of Shapley values: {phi.sum():.2f}  (pred - baseline = {prediction - baseline_pred:.2f})")

    print("Generating waterfall plot…")
    fig1 = plot_waterfall(phi, top_names, baseline_pred, prediction)
    save_figure(fig1, "ch17_shapley_values", "shapley_waterfall")

    print("Generating violin plot over test set…")
    # Use SHAP library for efficiency over the full test set
    import shap
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_test[:200])

    fig2 = go.Figure()
    for i, name in enumerate(feature_names):
        fig2.add_trace(go.Violin(
            y=shap_vals[:, i],
            name=name,
            box_visible=True,
            meanline_visible=True,
            line_color=COLORS["primary"],
        ))
    fig2.add_hline(y=0, line_dash="dot", line_color=COLORS["neutral"])
    fig2.update_layout(
        title="Shapley Values — Distribution Across Test Set",
        yaxis_title="Shapley value",
        showlegend=False,
    )
    save_figure(fig2, "ch17_shapley_values", "shapley_all_features")


if __name__ == "__main__":
    main()

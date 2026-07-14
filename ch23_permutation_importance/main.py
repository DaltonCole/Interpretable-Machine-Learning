"""
Chapter 23: Permutation Feature Importance
https://christophm.github.io/interpretable-ml-book/feature-importance.html

Permutation importance measures how much model performance drops when a feature's
values are randomly shuffled, breaking its relationship with the target.

Figures produced:
  1. perm_importance.html/png      — importance ± std over multiple shuffles
  2. perm_importance_ratio.html/png — ratio of shuffled / baseline RMSE per feature
"""
import numpy as np
import plotly.graph_objects as go
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error, root_mean_squared_error

from shared.datasets import load_regression
from shared.models import get_regression_model
from shared.theme import COLORS, save_figure

N_REPEATS = 10


def plot_importance_bars(result, feature_names: list) -> go.Figure:
    """Bar chart of mean importance ± 1 std, sorted descending."""
    means = result.importances_mean
    stds = result.importances_std
    order = np.argsort(means)

    fig = go.Figure(go.Bar(
        x=means[order],
        y=[feature_names[i] for i in order],
        orientation="h",
        error_x=dict(type="data", array=stds[order], visible=True),
        marker_color=[
            COLORS["primary"] if m > 0 else COLORS["neutral"]
            for m in means[order]
        ],
    ))
    fig.add_vline(x=0, line_color=COLORS["neutral"], line_width=1)
    fig.update_layout(
        title=f"Permutation Feature Importance (n_repeats={N_REPEATS})",
        xaxis_title="Mean decrease in R²",
        yaxis_title="Feature",
    )
    return fig


def plot_importance_ratio(model, X_test, y_test, feature_names: list) -> go.Figure:
    """Ratio of shuffled RMSE to baseline RMSE — shows proportional degradation."""
    baseline_rmse = root_mean_squared_error(y_test, model.predict(X_test))
    rng = np.random.default_rng(42)
    ratios, names = [], []
    for i, name in enumerate(feature_names):
        X_perm = X_test.copy()
        rng.shuffle(X_perm[:, i])
        perm_rmse = root_mean_squared_error(y_test, model.predict(X_perm))
        ratios.append(perm_rmse / baseline_rmse)
        names.append(name)

    order = np.argsort(ratios)
    fig = go.Figure(go.Bar(
        x=[ratios[i] for i in order],
        y=[names[i] for i in order],
        orientation="h",
        marker_color=[COLORS["primary"] if r > 1 else COLORS["neutral"]
                      for r in [ratios[i] for i in order]],
    ))
    fig.add_vline(x=1.0, line_dash="dash", line_color=COLORS["neutral"],
                  annotation_text="baseline RMSE")
    fig.update_layout(
        title="Permuted RMSE / Baseline RMSE (higher = more important)",
        xaxis_title="RMSE ratio",
    )
    return fig


def main():
    X_train, X_test, y_train, y_test, feature_names = load_regression()
    model = get_regression_model(X_train, y_train)

    print(f"Computing permutation importance ({N_REPEATS} repeats)…")
    result = permutation_importance(
        model, X_test, y_test, n_repeats=N_REPEATS, random_state=42, scoring="r2"
    )

    print("Generating importance bar chart…")
    fig1 = plot_importance_bars(result, feature_names)
    save_figure(fig1, "ch23_permutation_importance", "perm_importance")

    print("Generating RMSE ratio chart…")
    fig2 = plot_importance_ratio(model, X_test, y_test, feature_names)
    save_figure(fig2, "ch23_permutation_importance", "perm_importance_ratio")


if __name__ == "__main__":
    main()

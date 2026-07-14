"""
Chapter 8: GLM, GAM and More
https://christophm.github.io/interpretable-ml-book/extend-lm.html

Figures produced:
  1. gam_shape_functions.html/png — partial response (shape function) per feature
  2. gam_vs_linear.html/png       — GAM predictions vs. linear model predictions

Uses: pygam (LinearGAM) for the additive model.
"""
import operator
from functools import reduce

import numpy as np
import plotly.graph_objects as go
from pygam import LinearGAM, s
from sklearn.metrics import r2_score

from shared.datasets import load_regression
from shared.models import get_regression_model
from shared.theme import COLORS, save_figure


def plot_shape_functions(gam: LinearGAM, feature_names: list) -> go.Figure:
    """One subplot per feature showing its shape function (partial effect)."""
    n_features = len(feature_names)
    cols = 3
    rows = (n_features + cols - 1) // cols

    from plotly.subplots import make_subplots
    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=feature_names,
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
    )

    for i, name in enumerate(feature_names):
        XX = gam.generate_X_grid(term=i)
        pdep, confi = gam.partial_dependence(term=i, X=XX, width=0.95)

        row, col = divmod(i, cols)
        fig.add_trace(
            go.Scatter(x=XX[:, i], y=pdep, mode="lines",
                       line=dict(color=COLORS["primary"], width=2), name=name, showlegend=False),
            row=row + 1, col=col + 1,
        )
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([XX[:, i], XX[::-1, i]]),
                y=np.concatenate([confi[:, 0], confi[::-1, 1]]),
                fill="toself", fillcolor="rgba(37,99,235,0.15)",
                line=dict(color="rgba(0,0,0,0)"), showlegend=False,
            ),
            row=row + 1, col=col + 1,
        )

    fig.update_layout(
        title="GAM Shape Functions (partial effects per feature)",
        height=rows * 220,
    )
    return fig


def plot_gam_vs_linear(gam_preds, linear_preds, y_test, gam_r2: float, linear_r2: float) -> go.Figure:
    """Scatter of actual vs. predicted for GAM and linear model."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=y_test, y=gam_preds, mode="markers",
        marker=dict(color=COLORS["primary"], size=5, opacity=0.5),
        name=f"GAM (R²={gam_r2:.3f})",
    ))
    fig.add_trace(go.Scatter(
        x=y_test, y=linear_preds, mode="markers",
        marker=dict(color=COLORS["accent"], size=5, opacity=0.5),
        name=f"Linear (R²={linear_r2:.3f})",
    ))
    lim = [min(y_test.min(), gam_preds.min()), max(y_test.max(), gam_preds.max())]
    fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines",
                             line=dict(dash="dash", color=COLORS["neutral"]), name="Perfect fit"))
    fig.update_layout(
        title="GAM vs. Linear Regression — Actual vs. Predicted",
        xaxis_title="Actual rentals",
        yaxis_title="Predicted rentals",
    )
    return fig


def main():
    X_train, X_test, y_train, y_test, feature_names = load_regression()

    print("Fitting GAM…")
    # One spline term per feature
    gam = LinearGAM(reduce(operator.add, [s(i) for i in range(X_train.shape[1])])).fit(X_train, y_train)
    gam_preds = gam.predict(X_test)

    print("Fitting linear model for comparison…")
    linear = get_regression_model(X_train, y_train, model_type="linear")
    linear_preds = linear.predict(X_test)

    print("Generating shape function plot…")
    fig1 = plot_shape_functions(gam, feature_names)
    save_figure(fig1, "ch08_glm_gam", "gam_shape_functions")

    print("Generating GAM vs. linear comparison…")
    gam_r2 = r2_score(y_test, gam_preds)
    linear_r2 = r2_score(y_test, linear_preds)
    print(f"  GAM R²={gam_r2:.3f}  Linear R²={linear_r2:.3f}")
    fig2 = plot_gam_vs_linear(gam_preds, linear_preds, y_test, gam_r2, linear_r2)
    save_figure(fig2, "ch08_glm_gam", "gam_vs_linear")


if __name__ == "__main__":
    main()

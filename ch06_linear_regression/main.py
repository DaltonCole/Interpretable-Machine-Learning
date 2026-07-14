"""
Chapter 6: Linear Regression
https://christophm.github.io/interpretable-ml-book/limo.html

Figures produced:
  1. coefficients.html/png  — standardized weights with 95% confidence intervals
  2. effects.html/png       — distribution of weight × feature value per feature
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from shared.datasets import load_regression
from shared.theme import COLORS, save_figure


def plot_coefficients(coefs: np.ndarray, ci: np.ndarray, feature_names: list) -> go.Figure:
    """Weight plot: standardized coefficients with 95% CI, sorted by magnitude."""
    order = np.argsort(np.abs(coefs))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=coefs[order],
        y=[feature_names[i] for i in order],
        mode="markers",
        error_x=dict(type="data", array=ci[order], visible=True, color=COLORS["neutral"]),
        marker=dict(size=10, color=COLORS["primary"]),
        name="Coefficient",
    ))
    fig.add_vline(x=0, line_dash="dash", line_color=COLORS["neutral"], line_width=1)
    fig.update_layout(
        title="Linear Regression — Standardized Coefficients (95% CI)",
        xaxis_title="Weight",
        yaxis_title="Feature",
        height=500,
    )
    return fig


def plot_effects(coefs: np.ndarray, X_scaled: np.ndarray, feature_names: list) -> go.Figure:
    """Effect plot: distribution of weight × feature value across instances."""
    effects = X_scaled * coefs
    fig = go.Figure()
    for i, name in enumerate(feature_names):
        fig.add_trace(go.Box(
            x=effects[:, i],
            name=name,
            orientation="h",
            boxpoints=False,
            marker_color=COLORS["primary"],
        ))
    fig.update_layout(
        title="Feature Effects — weight × feature value",
        xaxis_title="Effect on predicted rental count",
        showlegend=False,
        height=max(400, len(feature_names) * 35),
    )
    return fig


def main():
    X_train, X_test, y_train, y_test, feature_names = load_regression()

    pipe = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))])
    pipe.fit(X_train, y_train)

    scaler: StandardScaler = pipe.named_steps["scaler"]
    model: Ridge = pipe.named_steps["ridge"]
    coefs = model.coef_

    # 95% CI via residual standard error (OLS approximation for Ridge)
    X_scaled = scaler.transform(X_train)
    y_pred = model.predict(X_scaled)
    residuals = y_train - y_pred
    n, p = X_scaled.shape
    rse = np.sqrt(np.sum(residuals ** 2) / (n - p - 1))
    se = rse * np.sqrt(np.diag(np.linalg.pinv(X_scaled.T @ X_scaled)))
    ci_95 = stats.t.ppf(0.975, df=n - p - 1) * se

    print("Generating coefficient plot…")
    fig1 = plot_coefficients(coefs, ci_95, feature_names)
    save_figure(fig1, "ch06_linear_regression", "coefficients")

    print("Generating effects plot…")
    X_test_scaled = scaler.transform(X_test)
    fig2 = plot_effects(coefs, X_test_scaled, feature_names)
    save_figure(fig2, "ch06_linear_regression", "effects")


if __name__ == "__main__":
    main()

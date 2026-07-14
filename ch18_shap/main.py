"""
Chapter 18: SHAP (SHapley Additive exPlanations)
https://christophm.github.io/interpretable-ml-book/shap.html

SHAP extends Shapley values with efficient algorithms (TreeSHAP, KernelSHAP)
and a rich suite of visualisations.

Figures produced:
  1. shap_waterfall.html/png   — force/waterfall for a single prediction
  2. shap_beeswarm.html/png    — summary beeswarm (importance + direction)
  3. shap_dependence.html/png  — dependence plot for the most important feature
"""
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import shap

from shared.datasets import load_regression
from shared.models import get_regression_model
from shared.theme import COLORS, save_figure

N_EXPLAIN = 500  # instances to explain for summary/dependence plots


def plot_waterfall(shap_values: np.ndarray, base_value: float,
                   prediction: float, feature_names: list) -> go.Figure:
    """Waterfall chart showing each feature's SHAP contribution."""
    order = np.argsort(np.abs(shap_values))[::-1][:10]  # top 10
    phi = shap_values[order]
    names = [feature_names[i] for i in order]

    running = base_value
    x_labels, y_values, bases, bar_colors = [], [], [], []
    for name, val in zip(names, phi):
        x_labels.append(name)
        y_values.append(val)
        bases.append(running)
        bar_colors.append(COLORS["positive"] if val >= 0 else COLORS["negative"])
        running += val

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=["E[f(x)]"],
        y=[base_value],
        marker_color=COLORS["neutral"],
        name="Base value",
    ))
    for lbl, val, base, color in zip(x_labels, y_values, bases, bar_colors):
        fig.add_trace(go.Bar(
            x=[lbl], y=[val], base=[base],
            marker_color=color, showlegend=False,
            hovertemplate=f"{lbl}: {val:+.2f}<extra></extra>",
        ))
    fig.add_trace(go.Bar(
        x=["f(x)"], y=[prediction],
        marker_color=COLORS["primary"], name="Prediction",
    ))
    fig.update_layout(
        title=f"SHAP Waterfall — base={base_value:.1f}, prediction={prediction:.1f}",
        yaxis_title="Predicted rental count",
        barmode="stack",
        showlegend=False,
    )
    return fig


def plot_beeswarm(shap_values: np.ndarray, X: np.ndarray, feature_names: list) -> go.Figure:
    """Summary beeswarm: each point is one instance; colour = feature value.

    Uses numeric y positions with per-feature jitter so points separate visually.
    """
    mean_abs = np.abs(shap_values).mean(axis=0)
    order = np.argsort(mean_abs)[::-1][:12]  # top 12, importance descending

    fig = go.Figure()
    rng = np.random.default_rng(42)
    tick_vals, tick_text = [], []

    for rank, fi in enumerate(order[::-1]):  # bottom-to-top on y-axis
        jitter = rng.uniform(-0.35, 0.35, len(shap_values))
        y_pos = np.full(len(shap_values), float(rank)) + jitter
        tick_vals.append(rank)
        tick_text.append(feature_names[fi])
        fig.add_trace(go.Scatter(
            x=shap_values[:, fi],
            y=y_pos,
            mode="markers",
            marker=dict(
                size=4,
                color=X[:, fi],
                colorscale="RdBu_r",
                opacity=0.6,
                showscale=(rank == 0),
                colorbar=dict(title="Feature value", len=0.4, y=0.2) if rank == 0 else None,
            ),
            showlegend=False,
            hovertemplate=f"{feature_names[fi]}: SHAP=%{{x:.2f}}<extra></extra>",
        ))

    fig.add_vline(x=0, line_dash="dot", line_color=COLORS["neutral"])
    fig.update_layout(
        title="SHAP Beeswarm — Feature Importance and Direction",
        xaxis_title="SHAP value (impact on prediction)",
        yaxis=dict(tickvals=tick_vals, ticktext=tick_text, showgrid=False),
        height=520,
    )
    return fig


def plot_dependence(shap_values: np.ndarray, X: np.ndarray,
                    feature_idx: int, interact_idx: int,
                    feature_names: list) -> go.Figure:
    """Scatter of feature value vs. SHAP value, coloured by an interaction feature."""
    fig = go.Figure(go.Scatter(
        x=X[:, feature_idx],
        y=shap_values[:, feature_idx],
        mode="markers",
        marker=dict(
            color=X[:, interact_idx],
            colorscale="Viridis",
            size=5,
            opacity=0.7,
            colorbar=dict(title=feature_names[interact_idx]),
        ),
        hovertemplate=(
            f"{feature_names[feature_idx]}=%{{x:.2f}}<br>"
            f"SHAP=%{{y:.2f}}<extra></extra>"
        ),
    ))
    fig.add_hline(y=0, line_dash="dot", line_color=COLORS["neutral"])
    fig.update_layout(
        title=f"SHAP Dependence — {feature_names[feature_idx]}",
        xaxis_title=feature_names[feature_idx],
        yaxis_title=f"SHAP value for {feature_names[feature_idx]}",
    )
    return fig


def main():
    X_train, X_test, y_train, y_test, feature_names = load_regression()
    model = get_regression_model(X_train, y_train)

    print("Computing SHAP values with TreeExplainer…")
    explainer = shap.TreeExplainer(model)
    X_explain = X_test[:N_EXPLAIN]
    shap_values = explainer.shap_values(X_explain)
    base_value = float(np.atleast_1d(explainer.expected_value)[0])

    instance_idx = 0
    pred = model.predict(X_explain[[instance_idx]])[0]

    print("Generating SHAP waterfall…")
    fig1 = plot_waterfall(shap_values[instance_idx], base_value, pred, feature_names)
    save_figure(fig1, "ch18_shap", "shap_waterfall")

    print("Generating SHAP beeswarm…")
    fig2 = plot_beeswarm(shap_values, X_explain, feature_names)
    save_figure(fig2, "ch18_shap", "shap_beeswarm")

    print("Generating SHAP dependence plot…")
    top_idx = int(np.argmax(np.abs(shap_values).mean(axis=0)))
    second_idx = int(np.argsort(np.abs(shap_values).mean(axis=0))[-2])
    fig3 = plot_dependence(shap_values, X_explain, top_idx, second_idx, feature_names)
    save_figure(fig3, "ch18_shap", "shap_dependence")


if __name__ == "__main__":
    main()

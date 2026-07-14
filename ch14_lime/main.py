"""
Chapter 14: LIME — Local Interpretable Model-Agnostic Explanations
https://christophm.github.io/interpretable-ml-book/lime.html

Figures produced:
  1. lime_explanation.html/png — feature contributions for a single prediction
  2. lime_stability.html/png   — explanation variance across repeated runs (stability check)

Uses: lime.lime_tabular.LimeTabularExplainer
"""
import numpy as np
import plotly.graph_objects as go
from lime.lime_tabular import LimeTabularExplainer

from shared.datasets import load_regression
from shared.models import get_regression_model
from shared.theme import COLORS, save_figure

N_SAMPLES = 1000   # LIME neighbourhood size
N_FEATURES = 10    # features to show in the explanation
N_STABILITY = 10   # repeated explanations for stability plot


def plot_lime_explanation(exp_list: list, feature_names: list, pred_value: float) -> go.Figure:
    """Horizontal bar chart of LIME feature contributions."""
    features = [f for f, _ in exp_list]
    weights = [w for _, w in exp_list]
    colors = [COLORS["positive"] if w > 0 else COLORS["negative"] for w in weights]

    fig = go.Figure(go.Bar(
        x=weights,
        y=features,
        orientation="h",
        marker_color=colors,
    ))
    fig.add_vline(x=0, line_color=COLORS["neutral"], line_width=1)
    fig.update_layout(
        title=f"LIME Explanation — predicted rental count: {pred_value:.0f}",
        xaxis_title="Feature contribution",
        yaxis_title="Feature condition",
        height=max(350, len(features) * 35),
    )
    return fig


def plot_lime_stability(weights_matrix: np.ndarray, feature_labels: list) -> go.Figure:
    """Box plot showing variance of each feature's weight across repeated LIME runs."""
    fig = go.Figure()
    for i, label in enumerate(feature_labels):
        fig.add_trace(go.Box(
            y=weights_matrix[:, i],
            name=label,
            boxpoints="all",
            jitter=0.3,
            marker=dict(size=4),
            line=dict(color=COLORS["primary"]),
        ))
    fig.add_hline(y=0, line_dash="dot", line_color=COLORS["neutral"])
    fig.update_layout(
        title="LIME Stability — Weight Distribution Across Repeated Explanations",
        yaxis_title="Feature weight",
        xaxis_title="Feature condition",
        showlegend=False,
    )
    return fig


def main():
    X_train, X_test, y_train, y_test, feature_names = load_regression()
    model = get_regression_model(X_train, y_train)

    explainer = LimeTabularExplainer(
        training_data=X_train,
        feature_names=feature_names,
        mode="regression",
        random_state=42,
    )

    instance = X_test[0]
    pred = model.predict(instance.reshape(1, -1))[0]

    print("Computing LIME explanation…")
    exp = explainer.explain_instance(
        instance, model.predict, num_features=N_FEATURES, num_samples=N_SAMPLES
    )
    fig1 = plot_lime_explanation(exp.as_list(), feature_names, pred)
    save_figure(fig1, "ch14_lime", "lime_explanation")

    print(f"Running {N_STABILITY} repeated explanations for stability…")
    top_features = [f for f, _ in exp.as_list()]

    all_weights = []
    for seed in range(N_STABILITY):
        exp_i = LimeTabularExplainer(
            X_train, feature_names=feature_names, mode="regression", random_state=seed
        ).explain_instance(instance, model.predict, num_features=N_FEATURES, num_samples=N_SAMPLES)
        weight_map = dict(exp_i.as_list())
        all_weights.append([weight_map.get(f, 0.0) for f in top_features])

    weights_matrix = np.array(all_weights)
    fig2 = plot_lime_stability(weights_matrix, top_features)
    save_figure(fig2, "ch14_lime", "lime_stability")


if __name__ == "__main__":
    main()

"""
Chapter 25: Global Surrogate Models
https://christophm.github.io/interpretable-ml-book/global.html

A global surrogate is an interpretable model (e.g. decision tree) trained to
approximate a black-box model's predictions across the entire input space.

Figures produced:
  1. surrogate_tree.html/png    — decision tree surrogate visualised as sunburst
  2. surrogate_fidelity.html/png — scatter of black-box vs. surrogate predictions (R²)
"""
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor, _tree

from shared.datasets import load_regression
from shared.models import get_regression_model
from shared.theme import COLORS, save_figure


def tree_to_sunburst(tree, feature_names):
    """Convert a fitted DecisionTreeRegressor to Plotly sunburst data."""
    t = tree.tree_
    ids, labels, parents, values = [], [], [], []

    def recurse(node, parent_label):
        if t.feature[node] != _tree.TREE_UNDEFINED:
            feat = feature_names[t.feature[node]]
            thresh = t.threshold[node]
            label = f"{feat}<br>≤{thresh:.2f}"
        else:
            label = f"Leaf<br>{t.value[node][0][0]:.0f}"
        ids.append(f"node{node}")
        labels.append(label)
        parents.append(parent_label)
        values.append(int(t.n_node_samples[node]))
        if t.children_left[node] != _tree.TREE_LEAF:
            recurse(t.children_left[node], f"node{node}")
            recurse(t.children_right[node], f"node{node}")

    recurse(0, "")
    return ids, labels, parents, values


def plot_fidelity(bb_preds: np.ndarray, surrogate_preds: np.ndarray,
                  r2: float) -> go.Figure:
    lim = [min(bb_preds.min(), surrogate_preds.min()),
           max(bb_preds.max(), surrogate_preds.max())]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=bb_preds, y=surrogate_preds, mode="markers",
        marker=dict(color=COLORS["primary"], size=4, opacity=0.5),
        name="Predictions",
    ))
    fig.add_trace(go.Scatter(
        x=lim, y=lim, mode="lines",
        line=dict(dash="dash", color=COLORS["neutral"]),
        name="Perfect fidelity",
    ))
    fig.update_layout(
        title=f"Surrogate Fidelity — R² = {r2:.3f}",
        xaxis_title="Black-box prediction",
        yaxis_title="Surrogate prediction",
    )
    return fig


def main():
    X_train, X_test, y_train, y_test, feature_names = load_regression()

    print("Training black-box model…")
    blackbox = get_regression_model(X_train, y_train)  # random forest
    bb_train_preds = blackbox.predict(X_train)
    bb_test_preds = blackbox.predict(X_test)

    print("Training decision-tree surrogate on black-box outputs…")
    surrogate = DecisionTreeRegressor(max_depth=4, random_state=42)
    surrogate.fit(X_train, bb_train_preds)
    surrogate_test_preds = surrogate.predict(X_test)

    r2 = r2_score(bb_test_preds, surrogate_test_preds)
    print(f"  Surrogate fidelity R² = {r2:.3f}")

    print("Generating surrogate tree visualisation…")
    ids, labels, parents, values = tree_to_sunburst(surrogate, feature_names)
    fig1 = go.Figure(go.Sunburst(
        ids=ids, labels=labels, parents=parents, values=values,
        branchvalues="total",
        hovertemplate="<b>%{label}</b><br>Samples: %{value}<extra></extra>",
    ))
    fig1.update_layout(title=f"Global Surrogate Decision Tree (max_depth=4, R²={r2:.3f})")
    save_figure(fig1, "ch25_surrogate_models", "surrogate_tree")

    print("Generating fidelity scatter…")
    fig2 = plot_fidelity(bb_test_preds, surrogate_test_preds, r2)
    save_figure(fig2, "ch25_surrogate_models", "surrogate_fidelity")


if __name__ == "__main__":
    main()

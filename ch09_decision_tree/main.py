"""
Chapter 9: Decision Tree
https://christophm.github.io/interpretable-ml-book/tree.html

Figures produced:
  1. tree_structure.html/png   — interactive tree diagram (Plotly sunburst)
  2. feature_importance.html/png — impurity-based feature importances
"""
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.tree import DecisionTreeRegressor, _tree

from shared.datasets import load_regression
from shared.theme import COLORS, save_figure


def tree_to_sunburst(tree: DecisionTreeRegressor, feature_names: list):
    """Extract tree structure into lists for a Plotly sunburst chart."""
    t = tree.tree_
    ids, labels, parents, values = [], [], [], []

    def recurse(node_id, parent_id):
        if t.feature[node_id] != _tree.TREE_UNDEFINED:
            feat = feature_names[t.feature[node_id]]
            threshold = t.threshold[node_id]
            label = f"{feat}<br>≤ {threshold:.2f}"
        else:
            label = f"Leaf<br>{t.value[node_id][0][0]:.1f}"

        ids.append(f"node{node_id}")
        labels.append(label)
        parents.append(f"node{parent_id}" if parent_id is not None else "")
        values.append(int(t.n_node_samples[node_id]))

        if t.children_left[node_id] != _tree.TREE_LEAF:
            recurse(t.children_left[node_id], node_id)
            recurse(t.children_right[node_id], node_id)

    recurse(0, None)
    return ids, labels, parents, values


def plot_tree_sunburst(tree, feature_names) -> go.Figure:
    ids, labels, parents, values = tree_to_sunburst(tree, feature_names)
    fig = go.Figure(go.Sunburst(
        ids=ids,
        labels=labels,
        parents=parents,
        values=values,
        branchvalues="total",
        hovertemplate="<b>%{label}</b><br>Samples: %{value}<extra></extra>",
    ))
    fig.update_layout(title="Decision Tree Structure (node size = sample count)")
    return fig


def plot_feature_importance(importances, feature_names) -> go.Figure:
    order = np.argsort(importances)
    fig = go.Figure(go.Bar(
        x=importances[order],
        y=[feature_names[i] for i in order],
        orientation="h",
        marker_color=COLORS["primary"],
    ))
    fig.update_layout(
        title="Decision Tree — Impurity-Based Feature Importance",
        xaxis_title="Mean Decrease in Impurity",
        yaxis_title="Feature",
    )
    return fig


def main():
    X_train, X_test, y_train, y_test, feature_names = load_regression()

    tree = DecisionTreeRegressor(max_depth=4, random_state=42)
    tree.fit(X_train, y_train)

    print("Generating tree structure…")
    fig1 = plot_tree_sunburst(tree, feature_names)
    save_figure(fig1, "ch09_decision_tree", "tree_structure")

    print("Generating feature importance…")
    fig2 = plot_feature_importance(tree.feature_importances_, feature_names)
    save_figure(fig2, "ch09_decision_tree", "feature_importance")


if __name__ == "__main__":
    main()

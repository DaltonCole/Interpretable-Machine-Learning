"""
Chapter 10: Decision Rules
https://christophm.github.io/interpretable-ml-book/rules.html

Figures produced:
  1. rules_table.html/png     — extracted rules visualized as a heatmap/table
  2. rule_coverage.html/png   — coverage vs. precision scatter per rule

Uses: sklearn DecisionTreeClassifier to extract IF-THEN rules.
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.tree import DecisionTreeClassifier, _tree

from shared.datasets import load_classification
from shared.theme import COLORS, save_figure


def extract_rules(tree: DecisionTreeClassifier, feature_names: list, class_names: list):
    """Recursively extract IF-THEN rules from a fitted decision tree."""
    t = tree.tree_
    rules = []

    def recurse(node, conditions):
        if t.feature[node] != _tree.TREE_UNDEFINED:
            feat = feature_names[t.feature[node]]
            thresh = t.threshold[node]
            recurse(t.children_left[node], conditions + [f"{feat} ≤ {thresh:.3f}"])
            recurse(t.children_right[node], conditions + [f"{feat} > {thresh:.3f}"])
        else:
            values = t.value[node][0]
            total = values.sum()
            predicted_class = class_names[np.argmax(values)]
            confidence = values.max() / total
            coverage = total
            rules.append({
                "rule": " AND ".join(conditions) if conditions else "True",
                "prediction": predicted_class,
                "confidence": round(confidence, 3),
                "coverage": int(coverage),
            })

    recurse(0, [])
    return pd.DataFrame(rules)


def plot_rule_coverage(df: pd.DataFrame) -> go.Figure:
    """Scatter of rule coverage vs. confidence, sized by coverage."""
    fig = px.scatter(
        df,
        x="coverage",
        y="confidence",
        color="prediction",
        size="coverage",
        hover_data=["rule"],
        title="Decision Rules — Coverage vs. Confidence",
        labels={"coverage": "Coverage (# instances)", "confidence": "Confidence"},
        color_discrete_sequence=COLORS["palette"],
    )
    fig.add_hline(y=0.5, line_dash="dash", line_color=COLORS["neutral"])
    return fig


def main():
    X_train, X_test, y_train, y_test, feature_names, class_names = load_classification()

    tree = DecisionTreeClassifier(max_depth=3, random_state=42)
    tree.fit(X_train, y_train)

    rules_df = extract_rules(tree, list(feature_names), list(class_names))
    print(f"Extracted {len(rules_df)} rules")

    print("Generating rules table…")
    row_colors = ["#F0F4FF" if i % 2 == 0 else "white" for i in range(len(rules_df))]
    fig1 = go.Figure(data=[go.Table(
        header=dict(
            values=["Rule", "Prediction", "Confidence", "Coverage"],
            fill_color="#2563EB",
            font=dict(color="white", size=13),
            align="left",
            height=36,
        ),
        cells=dict(
            values=[rules_df.rule, rules_df.prediction, rules_df.confidence, rules_df.coverage],
            fill_color=[row_colors] * 4,
            align="left",
            font=dict(size=12),
            height=30,
        ),
    )])
    fig1.update_layout(title="Extracted Decision Rules", height=max(400, len(rules_df) * 38))
    save_figure(fig1, "ch10_decision_rules", "rules_table")

    print("Generating coverage vs. confidence scatter…")
    fig2 = plot_rule_coverage(rules_df)
    save_figure(fig2, "ch10_decision_rules", "rule_coverage")


if __name__ == "__main__":
    main()

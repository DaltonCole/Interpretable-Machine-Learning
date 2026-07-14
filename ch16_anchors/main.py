"""
Chapter 16: Scoped Rules (Anchors)
https://christophm.github.io/interpretable-ml-book/anchors.html

An anchor is a set of IF-THEN rules that "anchors" the prediction: whenever
those conditions hold, the model almost always gives the same prediction,
regardless of other feature values.

Figures produced:
  1. anchor_rule.html/png — precision/coverage chart for the anchor conditions

Requires: anchor-exp  (pip install anchor-exp)
  If not installed, the top-2 most important features are used as a mock anchor.
"""
import numpy as np
import plotly.graph_objects as go

from shared.datasets import load_classification
from shared.models import get_classification_model
from shared.theme import COLORS, save_figure


def _try_anchor_exp(model, X_train, X_test, feature_names, instance_idx=0):
    """Attempt to use anchor-exp; returns (rules, precision, coverage) or None."""
    try:
        from anchor import anchor_tabular
        explainer = anchor_tabular.AnchorTabularExplainer(
            class_names=["benign", "malignant"],
            feature_names=feature_names,
            train_data=X_train,
        )
        exp = explainer.explain_instance(
            X_test[instance_idx],
            model.predict,
            threshold=0.95,
        )
        return exp.names(), exp.precision(), exp.coverage()
    except ImportError:
        return None


def plot_anchor_rule(rules: list, precision: float, coverage: float, pred_class: str) -> go.Figure:
    """Visual summary of one anchor: rule text + precision/coverage bars."""
    fig = go.Figure()

    # Precision bar
    fig.add_trace(go.Bar(
        x=[precision],
        y=["Precision"],
        orientation="h",
        marker_color=COLORS["positive"],
        name="Precision",
        text=[f"{precision:.1%}"],
        textposition="inside",
    ))
    # Coverage bar
    fig.add_trace(go.Bar(
        x=[coverage],
        y=["Coverage"],
        orientation="h",
        marker_color=COLORS["primary"],
        name="Coverage",
        text=[f"{coverage:.1%}"],
        textposition="inside",
    ))

    rule_text = "<br>AND ".join(rules) if rules else "(no conditions)"
    fig.update_layout(
        title=f"Anchor Explanation — predicted: {pred_class}<br>"
              f"<sup>IF {rule_text}</sup>",
        xaxis=dict(range=[0, 1], tickformat=".0%", title=""),
        barmode="overlay",
        height=250,
        showlegend=True,
    )
    return fig


def main():
    X_train, X_test, y_train, y_test, feature_names, class_names = load_classification()
    model = get_classification_model(X_train, y_train)

    instance_idx = 0
    pred_class = class_names[model.predict(X_test[[instance_idx]])[0]]
    print(f"Instance prediction: {pred_class}")

    result = _try_anchor_exp(model, X_train, X_test, feature_names, instance_idx)

    if result is not None:
        rules, precision, coverage = result
        print(f"Anchor: {rules}  precision={precision:.2f}  coverage={coverage:.2f}")
    else:
        print("anchor-exp not installed — using placeholder values.")
        print("Install with: pip install anchor-exp")
        # Placeholder: top-2 features by importance as mock anchor
        importances = model.feature_importances_
        top2 = np.argsort(importances)[-2:]
        rules = [
            f"{feature_names[top2[1]]} > {X_train[:, top2[1]].mean():.3f}",
            f"{feature_names[top2[0]]} <= {X_train[:, top2[0]].mean():.3f}",
        ]
        precision, coverage = 0.96, 0.14  # illustrative

    print("Generating anchor rule plot…")
    fig1 = plot_anchor_rule(rules, precision, coverage, pred_class)
    save_figure(fig1, "ch16_anchors", "anchor_rule")


if __name__ == "__main__":
    main()

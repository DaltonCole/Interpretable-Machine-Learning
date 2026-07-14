"""
Chapter 11: RuleFit
https://christophm.github.io/interpretable-ml-book/rulefit.html

Figures produced:
  1. rule_importances.html/png  — top rules and linear terms by importance
  2. rule_effects.html/png      — effect (weight × support) per rule

Uses: imodels.RuleFitRegressor
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from imodels import RuleFitRegressor

from shared.datasets import load_regression
from shared.theme import COLORS, save_figure


def _truncate(text: str, max_len: int = 60) -> str:
    return text if len(text) <= max_len else text[:max_len] + "…"


def plot_rule_importances(rules_df: pd.DataFrame, top_n: int = 20) -> go.Figure:
    """Horizontal bar chart of the most important rules."""
    top = rules_df.nlargest(top_n, "importance")
    fig = go.Figure(go.Bar(
        x=top["importance"].values,
        y=[_truncate(r) for r in top["rule"].values],
        orientation="h",
        marker_color=[COLORS["primary"] if t == "rule" else COLORS["accent"]
                      for t in top["type"].values],
        hovertext=top["rule"].values,
        hovertemplate="%{hovertext}<br>Importance: %{x:.4f}<extra></extra>",
    ))
    fig.update_layout(
        title=f"RuleFit — Top {top_n} Rules and Linear Terms by Importance",
        xaxis_title="Importance",
        yaxis_title="Rule / Feature",
        height=max(400, top_n * 28),
        showlegend=False,
    )
    return fig


def plot_rule_effects(rules_df: pd.DataFrame, top_n: int = 20) -> go.Figure:
    """Scatter of rule support vs. coefficient (positive = increases prediction)."""
    top = rules_df.nlargest(top_n, "importance")
    fig = go.Figure(go.Scatter(
        x=top["support"].values,
        y=top["coef"].values,
        mode="markers+text",
        text=top.index.astype(str),
        textposition="top center",
        marker=dict(
            size=top["importance"].values * 300,
            color=[COLORS["positive"] if c > 0 else COLORS["negative"] for c in top["coef"].values],
            opacity=0.7,
        ),
        hovertext=top["rule"].values,
    ))
    fig.add_hline(y=0, line_dash="dash", line_color=COLORS["neutral"])
    fig.update_layout(
        title="RuleFit — Support vs. Coefficient (size = importance)",
        xaxis_title="Rule Support (fraction of instances covered)",
        yaxis_title="Coefficient",
    )
    return fig


def main():
    X_train, X_test, y_train, y_test, feature_names = load_regression()

    print("Fitting RuleFit (this may take a moment)…")
    model = RuleFitRegressor(random_state=42)
    model.fit(X_train, y_train, feature_names=feature_names)

    rules_df = model._get_rules()
    rules_df = rules_df[rules_df["coef"] != 0].reset_index(drop=True)
    rules_df["importance"] = np.abs(rules_df["coef"]) * rules_df["support"]
    print(f"  {len(rules_df)} non-zero rules/terms")

    print("Generating importance plot…")
    fig1 = plot_rule_importances(rules_df)
    save_figure(fig1, "ch11_rulefit", "rule_importances")

    print("Generating effects plot…")
    fig2 = plot_rule_effects(rules_df)
    save_figure(fig2, "ch11_rulefit", "rule_effects")


if __name__ == "__main__":
    main()

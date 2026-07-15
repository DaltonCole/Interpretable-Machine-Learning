"""
Chapter 17: How Shapley Values Work
https://christophm.github.io/interpretable-ml-book/shapley.html

Conceptual figures explaining the Shapley value method — separate from the
model-output figures in main.py.

Figures produced:
  1. how_shapley_coalition.html/png  — grid heatmap of all 2^4 coalitions for top 4 features
  2. how_shapley_marginal.html/png   — marginal contribution of 'hr' across 8 coalitions
  3. how_shapley_properties.html/png — efficiency axiom: contributions sum to prediction gap
"""
import itertools
import math

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import shap

from shared.datasets import load_regression
from shared.models import get_regression_model
from shared.theme import COLORS, save_figure

CHAPTER = "ch17_shapley_values"
RNG = np.random.default_rng(42)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _predict_with_coalition(model, instance, coalition_global, all_global, baseline):
    """Return prediction when coalition features take instance values; rest at baseline."""
    x = baseline.copy()
    for gi in coalition_global:
        x[gi] = instance[gi]
    return float(model.predict(x.reshape(1, -1))[0])


# ── Figure 1: Coalition grid heatmap ─────────────────────────────────────────

def plot_shapley_coalition(model, X_train, feature_names) -> go.Figure:
    """Grid heatmap: rows = all 2^4 coalitions of top 4 features, cols = features.

    Blue = feature IN coalition, white = OUT.  Sorted by coalition size so the
    empty set (∅) is at top and the full set is at the bottom.
    """
    # Top 4 features by importance
    top4_global = list(np.argsort(model.feature_importances_)[-4:][::-1])
    top4_names = [feature_names[i] for i in top4_global]

    # All 16 coalitions sorted by size (0 → 4)
    coalitions = []
    for size in range(5):
        for combo in itertools.combinations(range(4), size):
            coalitions.append(list(combo))   # local indices within top4

    n_rows = len(coalitions)   # 16
    n_cols = 4

    z = np.zeros((n_rows, n_cols))
    text = []
    for r, coal in enumerate(coalitions):
        row_text = []
        for c in range(n_cols):
            if c in coal:
                z[r, c] = 1
                row_text.append("IN")
            else:
                row_text.append("OUT")
        text.append(row_text)

    row_labels = []
    for coal in coalitions:
        if len(coal) == 0:
            row_labels.append("∅")
        else:
            row_labels.append("{" + ", ".join(top4_names[c] for c in coal) + "}")

    # Colorscale: 0 = white (OUT), 1 = primary blue (IN)
    colorscale = [[0.0, "white"], [1.0, COLORS["primary"]]]

    fig = go.Figure(go.Heatmap(
        z=z,
        x=top4_names,
        y=row_labels,
        colorscale=colorscale,
        showscale=False,
        text=text,
        texttemplate="%{text}",
        textfont=dict(size=11, color="#1F2937"),
        xgap=2,
        ygap=2,
        hovertemplate="Coalition: %{y}<br>Feature: %{x}<br>Status: %{text}<extra></extra>",
    ))

    # Manual colour legend annotations
    fig.add_annotation(
        xref="paper", yref="paper", x=0.02, y=-0.06,
        text="■ IN coalition",
        showarrow=False, font=dict(size=12, color=COLORS["primary"]),
        xanchor="left",
    )
    fig.add_annotation(
        xref="paper", yref="paper", x=0.25, y=-0.06,
        text="□ OUT of coalition",
        showarrow=False, font=dict(size=12, color=COLORS["neutral"]),
        xanchor="left",
    )
    fig.add_annotation(
        xref="paper", yref="paper", x=0.5, y=-0.12,
        text="<b>Shapley value = weighted average marginal contribution across all coalitions</b>",
        showarrow=False, font=dict(size=12, color="#111827"),
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor=COLORS["primary"], borderwidth=1,
    )

    fig.update_layout(
        title=(
            "Shapley Values — Each Feature's Contribution Is Averaged Over All Coalitions<br>"
            f"<sup>All 2⁴ = 16 subsets of {{{', '.join(top4_names)}}}; sorted by coalition size</sup>"
        ),
        xaxis=dict(title="Feature", side="top", tickfont=dict(size=13)),
        yaxis=dict(title="Coalition (S)", autorange="reversed", tickfont=dict(size=11)),
        height=500,
        margin=dict(t=100, b=90),
    )
    return fig


# ── Figure 2: Marginal contribution of 'hr' ──────────────────────────────────

def plot_shapley_marginal(model, X_train, X_test, feature_names) -> go.Figure:
    """Bar chart: for each of the 8 subsets of {temp, hum, season}, how much does
    adding 'hr' change the prediction?  Weighted average = Shapley value of hr.
    """
    # Top 4 features by importance
    top4_global = list(np.argsort(model.feature_importances_)[-4:][::-1])
    top4_names = [feature_names[i] for i in top4_global]

    hr_idx_global = feature_names.index("hr")

    # Ensure hr is in top4; if not, force it in place of the 4th feature
    if hr_idx_global in top4_global:
        hr_local = top4_global.index(hr_idx_global)
    else:
        top4_global[-1] = hr_idx_global
        top4_names[-1] = "hr"
        hr_local = 3

    other_local = [i for i in range(4) if i != hr_local]
    other_global = [top4_global[i] for i in other_local]
    other_names = [top4_names[i] for i in other_local]

    # Pick a test instance with a high prediction (interesting values)
    preds = model.predict(X_test)
    inst_idx = int(np.argsort(preds)[-5])
    instance = X_test[inst_idx]
    baseline = X_train.mean(axis=0)

    n = 4   # number of features in our Shapley game

    marginals, weights, labels = [], [], []
    for size in range(4):   # subsets of the 3 "other" features: size = 0, 1, 2, 3
        for combo in itertools.combinations(range(3), size):
            coal_global = [other_global[i] for i in combo]
            label = ("∅" if len(combo) == 0 else
                     "{" + ", ".join(other_names[i] for i in combo) + "}")
            labels.append(label)

            pred_without = _predict_with_coalition(model, instance, coal_global, top4_global, baseline)
            pred_with = _predict_with_coalition(model, instance, coal_global + [hr_idx_global], top4_global, baseline)
            marginals.append(pred_with - pred_without)

            k = len(combo)
            weights.append(math.factorial(k) * math.factorial(n - k - 1) / math.factorial(n))

    phi_hr = sum(w * m for w, m in zip(weights, marginals))
    marginals = np.array(marginals)
    bar_colors = [COLORS["positive"] if v >= 0 else COLORS["negative"] for v in marginals]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels,
        y=marginals,
        marker_color=bar_colors,
        name="Marginal contribution",
        hovertemplate="Coalition: %{x}<br>Δ prediction: %{y:+.1f} rentals<extra></extra>",
        texttemplate="%{y:+.1f}",
        textposition="outside",
    ))

    fig.add_hline(
        y=phi_hr,
        line_dash="dash",
        line_color=COLORS["primary"],
        line_width=2.5,
        annotation_text=f"φ(hr) = {phi_hr:.1f}  ←  weighted average",
        annotation_position="top right",
        annotation_font=dict(color=COLORS["primary"], size=13),
    )

    # Dummy traces for legend
    fig.add_trace(go.Bar(x=[None], y=[None], marker_color=COLORS["positive"], name="Positive contribution"))
    fig.add_trace(go.Bar(x=[None], y=[None], marker_color=COLORS["negative"], name="Negative contribution"))

    fig.add_annotation(
        xref="paper", yref="paper", x=0.5, y=-0.22,
        text=(f"<b>Shapley value φ(hr) = {phi_hr:.1f}</b>"
              " — the weighted average over all 8 marginal contributions"),
        showarrow=False, font=dict(size=12),
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor=COLORS["primary"], borderwidth=1,
    )

    fig.update_layout(
        title=(
            "Shapley Values — Marginal Contribution of One Feature Across Coalitions<br>"
            f"<sup>For each subset of {{{', '.join(other_names)}}}: how much does adding 'hr' change the prediction?</sup>"
        ),
        xaxis_title="Coalition (subsets without hr)",
        yaxis_title="Marginal contribution (rentals)",
        legend=dict(orientation="h", y=-0.18),
        height=480,
        margin=dict(t=90, b=100),
    )
    return fig


# ── Figure 3: Efficiency axiom ────────────────────────────────────────────────

def plot_shapley_properties(model, X_test, feature_names) -> go.Figure:
    """Two-panel figure.

    Left:  mean |SHAP| per feature — global importance ranking.
    Right: waterfall for one instance showing E[f(x)] + Σφᵢ = f(x) exactly.
    """
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_test[:200])
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[0]
    base_value = float(np.atleast_1d(explainer.expected_value)[0])

    # ── Left panel data ──
    mean_abs = np.abs(shap_vals).mean(axis=0)
    feat_order_asc = np.argsort(mean_abs)          # ascending for horizontal bar
    sorted_names = [feature_names[i] for i in feat_order_asc]
    sorted_imp = mean_abs[feat_order_asc]

    # ── Right panel: pick high-prediction instance ──
    preds = model.predict(X_test[:200])
    inst_idx = int(np.argmax(preds))
    phi = shap_vals[inst_idx]
    pred = preds[inst_idx]

    order = np.argsort(np.abs(phi))[::-1][:10]
    phi_top = phi[order]
    names_top = [feature_names[i] for i in order]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Feature Importance (mean |SHAP|)", "Efficiency: Contributions Sum to Prediction"],
        column_widths=[0.42, 0.58],
        horizontal_spacing=0.12,
    )

    # Left: horizontal importance bars
    fig.add_trace(go.Bar(
        x=sorted_imp,
        y=sorted_names,
        orientation="h",
        marker_color=COLORS["primary"],
        name="Mean |SHAP|",
        hovertemplate="%{y}: %{x:.2f}<extra></extra>",
    ), row=1, col=1)
    fig.update_xaxes(title_text="Mean |SHAP value|", row=1, col=1)

    # Right: waterfall (stacked bars)
    # Base bar
    fig.add_trace(go.Bar(
        x=["E[f(x)]"], y=[base_value], base=[0],
        marker_color=COLORS["neutral"],
        name="Expected value",
        hovertemplate=f"Expected value = {base_value:.1f}<extra></extra>",
        texttemplate=f"{base_value:.0f}",
        textposition="outside",
    ), row=1, col=2)

    running = base_value
    for name, val in zip(names_top, phi_top):
        color = COLORS["positive"] if val >= 0 else COLORS["negative"]
        fig.add_trace(go.Bar(
            x=[name], y=[val], base=[running],
            marker_color=color,
            showlegend=False,
            hovertemplate=f"{name}: {val:+.1f}<extra></extra>",
            texttemplate=f"{val:+.0f}",
            textposition="outside",
        ), row=1, col=2)
        running += val

    # Remaining SHAP (features outside top 10)
    remainder = pred - running
    if abs(remainder) > 1.0:
        fig.add_trace(go.Bar(
            x=["others"], y=[remainder], base=[running],
            marker_color=COLORS["neutral"],
            showlegend=False,
            hovertemplate=f"Other features: {remainder:+.1f}<extra></extra>",
            texttemplate=f"{remainder:+.0f}",
            textposition="outside",
        ), row=1, col=2)
        running += remainder

    # Final prediction bar
    fig.add_trace(go.Bar(
        x=["f(x)"], y=[pred], base=[0],
        marker_color=COLORS["primary"],
        name="Prediction",
        hovertemplate=f"Prediction = {pred:.1f}<extra></extra>",
        texttemplate=f"{pred:.0f}",
        textposition="outside",
    ), row=1, col=2)

    fig.update_xaxes(tickangle=-30, row=1, col=2)
    fig.update_yaxes(title_text="Predicted rentals", row=1, col=2)

    # Efficiency annotation (right panel, paper coords)
    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.78, y=0.97,
        text=f"E[f(x)] + Σφᵢ = {base_value:.0f} + {pred - base_value:.0f} = {pred:.0f}",
        showarrow=False, font=dict(size=12, color=COLORS["primary"]),
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor=COLORS["primary"], borderwidth=1,
        xanchor="center",
    )

    # Legend entries for colours
    fig.add_trace(go.Bar(x=[None], y=[None], marker_color=COLORS["positive"], name="Increases prediction"))
    fig.add_trace(go.Bar(x=[None], y=[None], marker_color=COLORS["negative"], name="Decreases prediction"))

    fig.update_layout(
        title=(
            "Shapley Values — Efficiency: Contributions Sum to the Prediction Gap<br>"
            "<sup>Efficiency axiom: E[f(x)] + Σ φᵢ(x) = f(x) exactly — no unexplained residual</sup>"
        ),
        barmode="stack",
        height=520,
        legend=dict(orientation="h", y=-0.18),
        margin=dict(t=90, b=70),
    )
    return fig


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    X_train, X_test, y_train, y_test, feature_names = load_regression()
    model = get_regression_model(X_train, y_train)

    print("Figure 1: Coalition grid heatmap…")
    fig1 = plot_shapley_coalition(model, X_train, feature_names)
    save_figure(fig1, CHAPTER, "how_shapley_coalition")

    print("Figure 2: Marginal contributions of hr…")
    fig2 = plot_shapley_marginal(model, X_train, X_test, feature_names)
    save_figure(fig2, CHAPTER, "how_shapley_marginal")

    print("Figure 3: Efficiency axiom (uses SHAP library)…")
    fig3 = plot_shapley_properties(model, X_test, feature_names)
    save_figure(fig3, CHAPTER, "how_shapley_properties")


if __name__ == "__main__":
    main()

"""
Chapter 18: How SHAP Works
https://christophm.github.io/interpretable-ml-book/shap.html

Conceptual figures explaining the SHAP (SHapley Additive exPlanations) method
— separate from the model-output figures in main.py.

Figures produced:
  1. how_shap_local.html/png       — waterfall for one prediction (local explanation)
  2. how_shap_global.html/png      — importance bar + beeswarm (global explanation)
  3. how_shap_interaction.html/png — dependence plot showing interaction via colour
"""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import shap

from shared.datasets import load_regression
from shared.models import get_regression_model
from shared.theme import COLORS, save_figure

CHAPTER = "ch18_shap"
RNG = np.random.default_rng(42)


# ── Figure 1: Waterfall for one instance ──────────────────────────────────────

def plot_shap_local(explainer, shap_values, X_test, model, feature_names) -> go.Figure:
    """Clean waterfall: base_value → SHAP contributions → prediction.

    Shows top-10 features by absolute SHAP value for the instance with the
    highest predicted rental count (most features active).
    """
    base_value = float(np.atleast_1d(explainer.expected_value)[0])

    # Pick instance with highest prediction for interesting contributions
    preds = model.predict(X_test[:300])
    inst_idx = int(np.argmax(preds))
    phi = shap_values[inst_idx]
    pred = preds[inst_idx]

    # Top 10 by |SHAP|
    order = np.argsort(np.abs(phi))[::-1][:10]
    phi_top = phi[order]
    names_top = [feature_names[i] for i in order]

    hr_idx = feature_names.index("hr")
    hr_shap = phi[hr_idx]

    # Build waterfall bars
    x_labels = ["E[f(x)]"]
    y_vals = [base_value]
    bases = [0.0]
    bar_colors = [COLORS["neutral"]]
    hover_texts = [f"Expected value (average prediction) = {base_value:.1f} rentals"]

    running = base_value
    for name, val in zip(names_top, phi_top):
        x_labels.append(name)
        y_vals.append(val)
        bases.append(running)
        bar_colors.append(COLORS["positive"] if val >= 0 else COLORS["negative"])
        hover_texts.append(f"<b>{name}</b>: SHAP = {val:+.1f} rentals")
        running += val

    x_labels.append("f(x)")
    y_vals.append(pred)
    bases.append(0.0)
    bar_colors.append(COLORS["primary"])
    hover_texts.append(f"Final prediction = {pred:.1f} rentals")

    fig = go.Figure()

    for lbl, val, base, col, htxt in zip(x_labels[:-1], y_vals[:-1], bases[:-1],
                                          bar_colors[:-1], hover_texts[:-1]):
        fig.add_trace(go.Bar(
            x=[lbl], y=[val], base=[base],
            marker_color=col,
            showlegend=False,
            hovertemplate=htxt + "<extra></extra>",
            texttemplate=f"{val:+.0f}" if lbl not in ("E[f(x)]",) else f"{val:.0f}",
            textposition="outside",
        ))

    fig.add_trace(go.Bar(
        x=["f(x)"], y=[pred], base=[0],
        marker_color=COLORS["primary"],
        name="Prediction",
        hovertemplate=hover_texts[-1] + "<extra></extra>",
        texttemplate=f"{pred:.0f}",
        textposition="outside",
    ))

    fig.add_hline(y=base_value, line_dash="dot", line_color=COLORS["neutral"], line_width=1)

    # Annotation about hr's contribution
    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.97, y=0.97,
        text=(f"<b>SHAP(hr) = {hr_shap:+.1f} rentals</b><br>"
              f"The model predicts {abs(hr_shap):.0f} more rentals<br>"
              f"because this hour is {'later' if hr_shap > 0 else 'earlier'} than average"),
        showarrow=False, align="right",
        font=dict(size=11, color=COLORS["primary"]),
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor=COLORS["primary"], borderwidth=1,
        xanchor="right", yanchor="top",
    )

    # Legend entries for colours
    fig.add_trace(go.Bar(x=[None], y=[None], marker_color=COLORS["positive"], name="Increases prediction"))
    fig.add_trace(go.Bar(x=[None], y=[None], marker_color=COLORS["negative"], name="Decreases prediction"))
    fig.add_trace(go.Bar(x=[None], y=[None], marker_color=COLORS["neutral"],  name="Expected value"))

    fig.update_layout(
        title=(
            "SHAP — Each Feature's Exact Contribution to One Prediction<br>"
            f"<sup>Instance predicted at {pred:.0f} rentals vs expected {base_value:.0f} | "
            "top 10 features by |SHAP value|</sup>"
        ),
        yaxis_title="Rental count",
        barmode="stack",
        xaxis_tickangle=-25,
        legend=dict(orientation="h", y=-0.18),
        height=520,
        margin=dict(t=90, b=80),
    )
    return fig


# ── Figure 2: Global importance + beeswarm ────────────────────────────────────

def plot_shap_global(shap_values, X_test_sub, feature_names) -> go.Figure:
    """Left: mean |SHAP| bar chart.  Right: beeswarm — one point per instance,
    colour = feature value (red=high, blue=low), x = SHAP value.

    Together they show BOTH the magnitude (left) and direction (right) of each
    feature's influence across the dataset.
    """
    n_features = len(feature_names)
    mean_abs = np.abs(shap_values).mean(axis=0)

    # Sort by importance (most important at top for horizontal bar = highest y value)
    order_desc = np.argsort(mean_abs)[::-1]   # descending, index 0 = most important
    order_asc = order_desc[::-1]               # ascending for horizontal bar (bottom = least)

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Global Feature Importance", "Beeswarm: Direction & Magnitude"],
        column_widths=[0.38, 0.62],
        horizontal_spacing=0.10,
    )

    # Left: horizontal importance bars (most important at top → ascending order for hbar)
    fig.add_trace(go.Bar(
        x=mean_abs[order_asc],
        y=[feature_names[i] for i in order_asc],
        orientation="h",
        marker_color=COLORS["primary"],
        name="Mean |SHAP|",
        hovertemplate="%{y}: %{x:.2f}<extra></extra>",
    ), row=1, col=1)
    fig.update_xaxes(title_text="Mean |SHAP value|", row=1, col=1)
    fig.update_yaxes(tickfont=dict(size=11), row=1, col=1)

    # Right: beeswarm
    # Features shown bottom-to-top in descending importance (rank 0 = bottom = least)
    tick_vals, tick_text = [], []
    n_show = min(n_features, 12)
    show_order = order_desc[:n_show][::-1]   # least important first → bottom of y-axis

    for rank, fi in enumerate(show_order):
        fname = feature_names[fi]
        feat_vals = X_test_sub[:, fi]

        # Jitter to separate overlapping points
        jitter = RNG.uniform(-0.35, 0.35, len(shap_values))
        y_pos = np.full(len(shap_values), float(rank)) + jitter

        tick_vals.append(rank)
        tick_text.append(fname)

        # Show colorbar only on the first (bottom) trace to avoid clutter
        show_cb = rank == 0

        fig.add_trace(go.Scatter(
            x=shap_values[:, fi],
            y=y_pos,
            mode="markers",
            marker=dict(
                size=4,
                color=feat_vals,
                colorscale="RdBu_r",
                opacity=0.55,
                showscale=show_cb,
                colorbar=dict(
                    title="Feature<br>value",
                    len=0.5,
                    y=0.25,
                    x=1.02,
                    tickvals=[],
                    ticktext=[],
                ) if show_cb else None,
            ),
            showlegend=False,
            hovertemplate=f"{fname}: SHAP=%{{x:.2f}}<extra></extra>",
        ), row=1, col=2)

    fig.add_vline(x=0, line_dash="dot", line_color=COLORS["neutral"], row=1, col=2)

    fig.update_yaxes(
        tickvals=tick_vals, ticktext=tick_text,
        showgrid=False, tickfont=dict(size=11),
        row=1, col=2,
    )
    fig.update_xaxes(title_text="SHAP value (impact on prediction)", row=1, col=2)

    # Annotation: colour scale meaning
    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.62, y=0.03,
        text="Colour: <span style='color:#DC2626'>■ high</span> → "
             "<span style='color:#2563EB'>■ low</span> feature value",
        showarrow=False, font=dict(size=11),
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor=COLORS["neutral"], borderwidth=1,
        xanchor="left",
    )

    fig.update_layout(
        title=(
            "SHAP — Aggregate Individual Explanations into Global Importance<br>"
            "<sup>Left: ranking by mean |SHAP| — Right: beeswarm shows direction (red=high feature value)</sup>"
        ),
        height=520,
        margin=dict(t=90, b=50, r=80),
    )
    return fig


# ── Figure 3: Dependence plot (interaction via colour) ────────────────────────

def plot_shap_interaction(shap_values, X_test_sub, feature_names) -> go.Figure:
    """Scatter of hr value vs SHAP(hr), coloured by a second feature.

    The pattern reveals that hr's effect on bike rentals is not uniform:
    it differs depending on the season — a classic feature interaction.
    """
    hr_idx = feature_names.index("hr")
    # Second feature: use the one with highest interaction signal (pick season or yr)
    # Compute correlation between |residual from 1D PDP| and each other feature
    # For simplicity, use "yr" as specified; fall back to second-most-important feature
    second_name = "yr" if "yr" in feature_names else feature_names[1]
    second_idx = feature_names.index(second_name)

    hr_vals = X_test_sub[:, hr_idx]
    second_vals = X_test_sub[:, second_idx]
    hr_shap = shap_values[:, hr_idx]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=hr_vals,
        y=hr_shap,
        mode="markers",
        marker=dict(
            color=second_vals,
            colorscale="RdBu",
            size=6,
            opacity=0.65,
            showscale=False,   # avoid overlap; described via annotations
        ),
        hovertemplate=(
            f"hr=%{{x}}<br>SHAP(hr)=%{{y:.1f}}<br>{second_name}=%{{marker.color:.2f}}"
            "<extra></extra>"
        ),
        name="Test instances",
    ))

    fig.add_hline(y=0, line_dash="dot", line_color=COLORS["neutral"], line_width=1.5)

    # Annotate rush hour peaks
    fig.add_annotation(
        x=8, y=hr_shap[np.abs(hr_vals - 8) < 0.5].mean() if (np.abs(hr_vals - 8) < 0.5).any() else 50,
        text="Morning rush<br>(hr ≈ 8)",
        showarrow=True, arrowhead=2, ax=30, ay=-40,
        font=dict(size=11, color=COLORS["primary"]),
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor=COLORS["primary"], borderwidth=1,
    )
    fig.add_annotation(
        x=17, y=hr_shap[np.abs(hr_vals - 17) < 0.5].mean() if (np.abs(hr_vals - 17) < 0.5).any() else 100,
        text="Evening rush<br>(hr ≈ 17)",
        showarrow=True, arrowhead=2, ax=-40, ay=-40,
        font=dict(size=11, color=COLORS["accent"]),
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor=COLORS["accent"], borderwidth=1,
    )

    # Colour scale legend as annotation
    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.02, y=0.98,
        text=(f"Colour = <b>{second_name}</b> value<br>"
              "<span style='color:#DC2626'>■ red = high</span>  "
              "<span style='color:#2563EB'>■ blue = low</span>"),
        showarrow=False, align="left",
        font=dict(size=11),
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor=COLORS["neutral"], borderwidth=1,
        xanchor="left", yanchor="top",
    )

    fig.update_layout(
        title=(
            "SHAP — Dependence Plot: How One Feature's Effect Depends on Another<br>"
            f"<sup>Each point = one test instance | colour = {second_name} value | "
            "vertical spread at same hr = interaction effect</sup>"
        ),
        xaxis=dict(title="hr (hour of day)", tickmode="linear", tick0=0, dtick=2),
        yaxis_title="SHAP value of hr (impact on rentals)",
        legend=dict(orientation="h", y=-0.18),
        height=500,
        margin=dict(t=90, b=60),
    )
    return fig


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    X_train, X_test, y_train, y_test, feature_names = load_regression()
    model = get_regression_model(X_train, y_train)

    print("Computing SHAP values with TreeExplainer…")
    explainer = shap.TreeExplainer(model)
    X_sub = X_test[:300]
    shap_values = explainer.shap_values(X_sub)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    print("Figure 1: SHAP waterfall (local explanation)…")
    fig1 = plot_shap_local(explainer, shap_values, X_sub, model, feature_names)
    save_figure(fig1, CHAPTER, "how_shap_local")

    print("Figure 2: Importance bar + beeswarm (global explanation)…")
    fig2 = plot_shap_global(shap_values, X_sub, feature_names)
    save_figure(fig2, CHAPTER, "how_shap_global")

    print("Figure 3: Dependence plot (interaction via colour)…")
    fig3 = plot_shap_interaction(shap_values, X_sub, feature_names)
    save_figure(fig3, CHAPTER, "how_shap_interaction")


if __name__ == "__main__":
    main()

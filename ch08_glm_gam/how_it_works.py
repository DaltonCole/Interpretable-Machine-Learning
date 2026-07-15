"""
Chapter 8: How GAMs Work
https://christophm.github.io/interpretable-ml-book/extend-lm.html

Conceptual figures explaining the method — separate from the model-output
figures in main.py.

Figures produced:
  1. how_gam_shape.html/png     — GAM vs linear partial effect on hour-of-day
  2. how_gam_additive.html/png  — waterfall showing each shape function's contribution
  3. how_gam_vs_linear.html/png — actual vs predicted comparison (GAM better fit)
"""
import operator
from functools import reduce

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pygam import LinearGAM, s
from sklearn.metrics import r2_score

from shared.datasets import load_regression
from shared.models import get_regression_model
from shared.theme import COLORS, save_figure

CHAPTER = "ch08_glm_gam"
RNG = np.random.default_rng(42)


def fit_gam(X_train, y_train):
    gam = LinearGAM(reduce(operator.add, [s(i) for i in range(X_train.shape[1])]))
    gam.fit(X_train, y_train)
    return gam


# ── Figure 1: GAM vs linear on hour-of-day ───────────────────────────────────

def plot_gam_shape(gam, linear_pipe, X_train, feature_names) -> go.Figure:
    """Compare GAM shape function vs linear model partial effect for 'hr' (hour of day).

    The GAM captures rush-hour peaks; the linear model can only draw a straight line.
    """
    hr_idx = feature_names.index("hr")

    # GAM partial dependence on hr
    XX = gam.generate_X_grid(term=hr_idx, n=200)
    pdep, confi = gam.partial_dependence(term=hr_idx, X=XX, width=0.95)
    hr_grid = XX[:, hr_idx]

    # Linear model partial effect (vary hr, hold others at median)
    X_med = np.median(X_train, axis=0)
    hr_range = np.linspace(X_train[:, hr_idx].min(), X_train[:, hr_idx].max(), 200)
    X_sweep = np.tile(X_med, (200, 1))
    X_sweep[:, hr_idx] = hr_range
    linear_preds = linear_pipe.predict(X_sweep)

    # Centre both curves at zero for fair shape comparison
    gam_centered   = pdep - pdep.mean()
    linear_centered = linear_preds - linear_preds.mean()

    fig = go.Figure()

    # CI band for GAM
    fig.add_trace(go.Scatter(
        x=np.concatenate([hr_grid, hr_grid[::-1]]),
        y=np.concatenate([confi[:, 0] - pdep.mean(), (confi[::-1, 1] - pdep.mean())]),
        fill="toself",
        fillcolor="rgba(37,99,235,0.12)",
        line=dict(color="rgba(0,0,0,0)"),
        showlegend=True,
        name="GAM 95% CI",
        hoverinfo="skip",
    ))

    # GAM shape function
    fig.add_trace(go.Scatter(
        x=hr_grid, y=gam_centered,
        mode="lines",
        line=dict(color=COLORS["primary"], width=3),
        name="GAM (spline)",
        hovertemplate="hr=%{x:.1f}<br>Shape effect=%{y:.1f}<extra></extra>",
    ))

    # Linear partial effect
    fig.add_trace(go.Scatter(
        x=hr_range, y=linear_centered,
        mode="lines",
        line=dict(color=COLORS["accent"], width=2.5, dash="dash"),
        name="Linear model (straight line)",
        hovertemplate="hr=%{x:.1f}<br>Linear effect=%{y:.1f}<extra></extra>",
    ))

    # Annotate rush hour peaks
    peak_8  = int(np.argmin(np.abs(hr_grid - 8)))
    peak_17 = int(np.argmin(np.abs(hr_grid - 17)))
    for peak_pos, label, ax_off in [(peak_8, "Morning rush<br>(GAM captures)", -60),
                                     (peak_17, "Evening rush<br>(GAM captures)", 60)]:
        fig.add_annotation(
            x=hr_grid[peak_pos], y=gam_centered[peak_pos] + 10,
            text=label, showarrow=True, arrowhead=2,
            ax=ax_off, ay=-45,
            font=dict(size=11, color=COLORS["primary"]),
            bgcolor="rgba(255,255,255,0.9)", bordercolor=COLORS["primary"], borderwidth=1,
        )

    # Annotate linear limitation
    mid_x = hr_range[100]
    fig.add_annotation(
        x=mid_x, y=float(linear_centered[100]) - 60,
        text="Linear model misses<br>the bimodal peak",
        showarrow=True, arrowhead=2,
        ax=50, ay=40,
        font=dict(size=11, color=COLORS["accent"]),
        bgcolor="rgba(255,255,255,0.9)", bordercolor=COLORS["accent"], borderwidth=1,
    )

    fig.add_hline(y=0, line_dash="dot", line_color=COLORS["neutral"], line_width=1)

    fig.update_layout(
        title=("GAM — Each Feature Gets Its Own Flexible Shape Function<br>"
               "<sup>Hour-of-day partial effect: GAM spline bends to fit rush hours; "
               "linear model is constrained to a single slope.</sup>"),
        xaxis=dict(title="Hour of day", tickmode="linear", tick0=0, dtick=2),
        yaxis_title="Centred partial effect (bike rentals)",
        legend=dict(orientation="h", y=-0.18),
        height=500,
    )
    return fig


# ── Figure 2: GAM prediction as a sum of shape functions ─────────────────────

def plot_gam_additive(gam, X_test, y_test, feature_names) -> go.Figure:
    """Waterfall chart: intercept → each shape function contribution → GAM prediction."""
    # Pick instance with highest predicted value (most features active)
    preds_all = gam.predict(X_test)
    inst_idx  = int(np.argmax(preds_all))
    instance  = X_test[inst_idx]
    prediction = preds_all[inst_idx]

    intercept = float(gam.coef_[-1])  # GAM intercept is stored as last coef

    # Get each term's partial dependence at this instance
    n_terms = len(feature_names)
    contributions = []
    for i in range(n_terms):
        val = float(np.atleast_1d(gam.partial_dependence(term=i, X=instance.reshape(1, -1)))[0])
        contributions.append(float(val))
    contributions = np.array(contributions)

    # Top 8 by absolute contribution
    order = np.argsort(np.abs(contributions))[::-1][:8]
    contrib_top = contributions[order]
    names_top   = [feature_names[i] for i in order]
    vals_top    = [instance[i] for i in order]

    # Build waterfall
    running = intercept
    x_labels  = ["Intercept"]
    bar_vals   = [intercept]
    bar_bases  = [0]
    bar_colors = [COLORS["neutral"]]
    hover_texts = [f"GAM intercept (grand mean) = {intercept:.1f}"]

    for name, contrib, feat_val in zip(names_top, contrib_top, vals_top):
        x_labels.append(f"f({name})")
        bar_vals.append(contrib)
        bar_bases.append(running)
        bar_colors.append(COLORS["positive"] if contrib >= 0 else COLORS["negative"])
        hover_texts.append(
            f"<b>f({name})</b><br>{name} = {feat_val:.2f}<br>Shape contribution: {contrib:+.1f}"
        )
        running += contrib

    x_labels.append("Prediction")
    bar_vals.append(prediction)
    bar_bases.append(0)
    bar_colors.append(COLORS["primary"])
    hover_texts.append(f"GAM prediction = {prediction:.0f} rentals<br>Actual = {y_test[inst_idx]:.0f}")

    fig = go.Figure()

    for lbl, val, base, col, htxt in zip(x_labels[:-1], bar_vals[:-1], bar_bases[:-1],
                                          bar_colors[:-1], hover_texts[:-1]):
        tpl = f"{val:.0f}" if lbl == "Intercept" else f"{val:+.0f}"
        fig.add_trace(go.Bar(
            x=[lbl], y=[val], base=[base],
            marker_color=col, showlegend=False,
            hovertemplate=htxt + "<extra></extra>",
            texttemplate=tpl, textposition="outside",
        ))

    fig.add_trace(go.Bar(
        x=["Prediction"], y=[prediction], base=[0],
        marker_color=COLORS["primary"], showlegend=False,
        hovertemplate=hover_texts[-1] + "<extra></extra>",
        texttemplate=f"{prediction:.0f}", textposition="outside",
    ))

    fig.add_hline(y=intercept, line_dash="dash", line_color=COLORS["neutral"], line_width=1)

    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.99, y=0.99,
        text=f"ŷ = intercept + Σ f<sub>j</sub>(x<sub>j</sub>)<br>"
             f"= {intercept:.0f} + ({contrib_top.sum():.0f}) = <b>{prediction:.0f}</b>",
        showarrow=False, align="right",
        font=dict(size=12, color=COLORS["primary"]),
        bgcolor="rgba(255,255,255,0.9)", bordercolor=COLORS["primary"], borderwidth=1,
    )

    fig.add_trace(go.Bar(x=[None], y=[None], marker_color=COLORS["positive"], name="Increases prediction"))
    fig.add_trace(go.Bar(x=[None], y=[None], marker_color=COLORS["negative"], name="Decreases prediction"))
    fig.add_trace(go.Bar(x=[None], y=[None], marker_color=COLORS["neutral"],  name="Intercept (baseline)"))

    fig.update_layout(
        title=("GAM — The Prediction Is a Sum of Shape Functions<br>"
               f"<sup>ŷ = intercept + f(hr) + f(temp) + … &nbsp;|&nbsp; "
               f"Instance #{inst_idx}: predicted {prediction:.0f}, actual {y_test[inst_idx]:.0f} rentals</sup>"),
        yaxis_title="Rental count contribution",
        barmode="stack",
        xaxis_tickangle=-30,
        height=520,
        legend=dict(orientation="h", y=-0.22),
    )
    return fig


# ── Figure 3: GAM vs linear actual vs predicted ───────────────────────────────

def plot_gam_vs_linear(gam, linear_pipe, X_test, y_test) -> go.Figure:
    """Side-by-side actual vs predicted scatter for GAM and linear model."""
    gam_preds    = gam.predict(X_test)
    linear_preds = linear_pipe.predict(X_test)

    gam_r2    = r2_score(y_test, gam_preds)
    linear_r2 = r2_score(y_test, linear_preds)

    gam_resid    = np.abs(y_test - gam_preds)
    linear_resid = np.abs(y_test - linear_preds)

    cmax = float(np.percentile(np.concatenate([gam_resid, linear_resid]), 95))

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[
            f"GAM  (R² = {gam_r2:.3f})",
            f"Linear Model  (R² = {linear_r2:.3f})",
        ],
        horizontal_spacing=0.12,
    )

    lim_lo = min(y_test.min(), gam_preds.min(), linear_preds.min())
    lim_hi = max(y_test.max(), gam_preds.max(), linear_preds.max())
    lim = [lim_lo, lim_hi]

    for col_idx, (preds, resid, model_label, color) in enumerate([
        (gam_preds,    gam_resid,    "GAM",    COLORS["primary"]),
        (linear_preds, linear_resid, "Linear", COLORS["accent"]),
    ], start=1):
        fig.add_trace(go.Scatter(
            x=preds, y=y_test,
            mode="markers",
            marker=dict(
                color=resid,
                colorscale="RdYlGn_r",
                cmin=0, cmax=cmax,
                size=4, opacity=0.55,
                showscale=False,
            ),
            name=model_label,
            showlegend=False,
            hovertemplate="Predicted: %{x:.0f}<br>Actual: %{y:.0f}<extra></extra>",
        ), row=1, col=col_idx)

        fig.add_trace(go.Scatter(
            x=lim, y=lim, mode="lines",
            line=dict(dash="dash", color=COLORS["neutral"], width=1.5),
            name="Perfect fit",
            showlegend=(col_idx == 1),
        ), row=1, col=col_idx)

        fig.update_xaxes(title_text="Predicted rentals", row=1, col=col_idx)
        fig.update_yaxes(title_text="Actual rentals", row=1, col=col_idx)

        # Inline color legend
        fig.add_annotation(
            row=1, col=col_idx,
            xref=f"x{'' if col_idx==1 else col_idx} domain",
            yref=f"y{'' if col_idx==1 else col_idx} domain",
            x=0.03, y=0.97,
            text="● green = small error<br>● red = large error",
            showarrow=False, align="left", xanchor="left", yanchor="top",
            font=dict(size=11),
            bgcolor="rgba(255,255,255,0.85)", bordercolor="#D1D5DB", borderwidth=1,
        )

    fig.update_layout(
        title=("GAM vs Linear — Better Fit with Maintained Interpretability<br>"
               f"<sup>GAM R² = {gam_r2:.3f} vs Linear R² = {linear_r2:.3f} — "
               "flexible shape functions reduce systematic errors at low/high counts.</sup>"),
        height=520,
        legend=dict(orientation="h", y=-0.18),
    )
    return fig


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    X_train, X_test, y_train, y_test, feature_names = load_regression()

    print("Fitting GAM…")
    gam = fit_gam(X_train, y_train)

    print("Fitting linear model for comparison…")
    linear_pipe = get_regression_model(X_train, y_train, model_type="linear")

    print("Figure 1: GAM shape function vs linear on hour-of-day…")
    fig1 = plot_gam_shape(gam, linear_pipe, X_train, feature_names)
    save_figure(fig1, CHAPTER, "how_gam_shape")

    print("Figure 2: Additive decomposition waterfall…")
    fig2 = plot_gam_additive(gam, X_test, y_test, feature_names)
    save_figure(fig2, CHAPTER, "how_gam_additive")

    print("Figure 3: GAM vs linear actual vs predicted…")
    fig3 = plot_gam_vs_linear(gam, linear_pipe, X_test, y_test)
    save_figure(fig3, CHAPTER, "how_gam_vs_linear")


if __name__ == "__main__":
    main()

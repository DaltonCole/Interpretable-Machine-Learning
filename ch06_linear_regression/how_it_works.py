"""
Chapter 6: How Linear Regression Works
https://christophm.github.io/interpretable-ml-book/limo.html

Conceptual figures explaining the method — separate from the model-output
figures in main.py.

Figures produced:
  1. how_lm_fit.html/png          — partial effect of top feature + residuals;
                                    shows "the slope is the coefficient"
  2. how_lm_contributions.html/png — waterfall breakdown of one prediction into
                                    per-feature contributions; shows full transparency
  3. how_lm_diagnostics.html/png  — actual vs predicted + residual distribution;
                                    shows where linearity holds
"""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from shared.datasets import load_regression
from shared.theme import COLORS, save_figure

CHAPTER = "ch06_linear_regression"
RNG = np.random.default_rng(42)


def fit_model(X_train, y_train):
    pipe = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))])
    pipe.fit(X_train, y_train)
    return pipe


def partial_effect(pipe, X_train, feature_idx, n_points=120):
    """Vary one feature across its range; hold all others at their median."""
    X_med = np.median(X_train, axis=0)
    feat_range = np.linspace(X_train[:, feature_idx].min(),
                             X_train[:, feature_idx].max(), n_points)
    X_sweep = np.tile(X_med, (n_points, 1))
    X_sweep[:, feature_idx] = feat_range
    return feat_range, pipe.predict(X_sweep)


# ── Figure 1: "The slope is the coefficient" ─────────────────────────────────

def plot_lm_fit(pipe, X_train, y_train, feature_names) -> go.Figure:
    """Scatter of hour-of-day vs rentals showing:
      - raw data points (transparent)
      - binned hourly mean (actual trend, bimodal)
      - linear model partial-effect line (straight)
    The gap between the two lines illustrates what linear regression captures
    and what it misses (non-linearity).
    """
    scaler   = pipe.named_steps["scaler"]
    model    = pipe.named_steps["ridge"]
    hr_idx   = feature_names.index("hr")
    unstd_coefs = model.coef_ / scaler.scale_
    slope    = unstd_coefs[hr_idx]   # rentals per hour

    grid, line_preds = partial_effect(pipe, X_train, hr_idx)

    # Hourly binned means (shows actual non-linear pattern)
    hours      = X_train[:, hr_idx].astype(int)
    bin_means  = np.array([y_train[hours == h].mean() if (hours == h).any() else 0
                           for h in range(24)])
    bin_counts = np.array([(hours == h).sum() for h in range(24)])

    # Sample scatter (not too many points)
    samp = RNG.choice(len(X_train), size=800, replace=False)

    fig = go.Figure()

    # Raw scatter
    fig.add_trace(go.Scatter(
        x=X_train[samp, hr_idx],
        y=y_train[samp],
        mode="markers",
        marker=dict(color=COLORS["primary"], size=4, opacity=0.18),
        name="Individual hours",
        hovertemplate="hr=%{x}<br>Rentals=%{y}<extra></extra>",
    ))

    # Linear model partial-effect
    fig.add_trace(go.Scatter(
        x=grid, y=line_preds,
        mode="lines",
        line=dict(color=COLORS["accent"], width=3),
        name=f"Linear model  (β_hr = {slope:+.1f} rentals/hr)",
    ))

    # Actual binned hourly means — shows what the model can't capture
    fig.add_trace(go.Scatter(
        x=np.arange(24),
        y=bin_means,
        mode="lines+markers",
        line=dict(color=COLORS["positive"], width=2.5),
        marker=dict(size=8),
        name="Actual hourly mean",
        hovertemplate="hr=%{x}<br>Mean rentals=%{y:.0f}<extra></extra>",
    ))

    # Annotate the slope on the linear line
    mid = 12
    slope_label_y = float(pipe.predict(
        np.tile(np.median(X_train, axis=0).reshape(1, -1), (1, 1))
    )[0]) + slope * (mid - float(np.median(grid)))
    fig.add_annotation(
        x=mid, y=slope_label_y + 55,
        text=f"β<sub>hr</sub> = {slope:+.1f} rentals per hour<br>"
             f"<i>(linear model's single slope for hr)</i>",
        showarrow=True, arrowhead=2,
        ax=0, ay=-45,
        font=dict(size=13, color=COLORS["accent"]),
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor=COLORS["accent"], borderwidth=1,
    )

    # Annotate the rush-hour peaks the linear model misses
    for peak_hr, label in [(8, "Morning<br>rush"), (17, "Evening<br>rush")]:
        fig.add_annotation(
            x=peak_hr, y=bin_means[peak_hr] + 30,
            text=label, showarrow=True, arrowhead=2,
            ax=0, ay=-35,
            font=dict(size=11, color=COLORS["positive"]),
        )

    fig.update_layout(
        title=("Linear Regression — The Slope Is the Coefficient<br>"
               "<sup>A single slope per feature: the model captures the overall trend "
               "but cannot bend to match non-linear patterns like rush hours.</sup>"),
        xaxis=dict(title="Hour of day", tickmode="linear", tick0=0, dtick=2),
        yaxis_title="Bike rentals (cnt)",
        legend=dict(orientation="h", y=-0.18),
        height=520,
    )
    return fig


# ── Figure 2: "Every prediction is a sum of contributions" ───────────────────

def plot_lm_contributions(pipe, X_test, y_test, feature_names) -> go.Figure:
    """Waterfall chart decomposing one prediction into per-feature contributions.

    contribution_i = β_i × (x_i − μ_i) / σ_i    (in target units)
    intercept + Σ contributions = prediction
    """
    scaler = pipe.named_steps["scaler"]
    model  = pipe.named_steps["ridge"]

    # Pick the instance with the largest predicted value (most features active)
    preds_all = pipe.predict(X_test)
    inst_idx  = int(np.argmax(preds_all))
    instance  = X_test[inst_idx]
    prediction = preds_all[inst_idx]

    intercept     = float(model.intercept_)
    contributions = model.coef_ * scaler.transform(instance.reshape(1, -1))[0]

    # Sort by absolute contribution, top 10
    order = np.argsort(np.abs(contributions))[::-1][:10]
    contrib_top  = contributions[order]
    names_top    = [feature_names[i] for i in order]
    vals_top     = [instance[i] for i in order]

    # Build waterfall: intercept → each contribution → prediction
    running = intercept
    x_labels = ["Intercept (avg prediction)"]
    bar_vals, bar_bases, bar_colors, hover_texts = [], [], [], []

    # Intercept bar
    bar_vals.append(intercept)
    bar_bases.append(0)
    bar_colors.append(COLORS["neutral"])
    hover_texts.append(f"Average prediction when all features are at their mean<br>= {intercept:.1f} rentals")

    for name, val, feat_val in zip(names_top, contrib_top, vals_top):
        x_labels.append(name)
        bar_vals.append(val)
        bar_bases.append(running)
        bar_colors.append(COLORS["positive"] if val >= 0 else COLORS["negative"])
        hover_texts.append(
            f"<b>{name}</b> = {feat_val:.2f}<br>"
            f"Contribution: {val:+.1f} rentals"
        )
        running += val

    x_labels.append("Prediction")
    bar_vals.append(prediction)
    bar_bases.append(0)
    bar_colors.append(COLORS["primary"])
    hover_texts.append(f"Final prediction: {prediction:.0f} rentals")

    fig = go.Figure()

    # Intercept + contributions as stacked bars
    for i, (lbl, val, base, col, htxt) in enumerate(
        zip(x_labels[:-1], bar_vals[:-1], bar_bases[:-1], bar_colors[:-1], hover_texts[:-1])
    ):
        fig.add_trace(go.Bar(
            x=[lbl], y=[val], base=[base],
            marker_color=col,
            showlegend=False,
            hovertemplate=htxt + "<extra></extra>",
            texttemplate=f"{val:+.0f}" if i > 0 else f"{val:.0f}",
            textposition="outside",
        ))

    # Final prediction bar
    fig.add_trace(go.Bar(
        x=["Prediction"], y=[prediction], base=[0],
        marker_color=COLORS["primary"],
        showlegend=False,
        hovertemplate=hover_texts[-1] + "<extra></extra>",
        texttemplate=f"{prediction:.0f}",
        textposition="outside",
    ))

    fig.add_hline(y=intercept, line_dash="dash", line_color=COLORS["neutral"], line_width=1)

    # Legend entries
    fig.add_trace(go.Bar(x=[None], y=[None], marker_color=COLORS["positive"], name="Increases prediction"))
    fig.add_trace(go.Bar(x=[None], y=[None], marker_color=COLORS["negative"], name="Decreases prediction"))
    fig.add_trace(go.Bar(x=[None], y=[None], marker_color=COLORS["neutral"],  name="Intercept (baseline)"))

    fig.update_layout(
        title=("Linear Regression — Every Prediction Is a Sum of Feature Contributions<br>"
               "<sup>intercept + Σ (β<sub>i</sub> × x<sub>i</sub>) = ŷ &nbsp;|&nbsp; "
               f"This instance: predicted {prediction:.0f} rentals, actual {y_test[inst_idx]:.0f}</sup>"),
        yaxis_title="Rental count contribution",
        barmode="stack",
        xaxis_tickangle=-30,
        height=520,
        legend=dict(orientation="h", y=-0.2),
    )
    return fig


# ── Figure 3: "Does it fit? Check the residuals" ─────────────────────────────

def plot_lm_diagnostics(pipe, X_test, y_test) -> go.Figure:
    """Actual vs predicted scatter (left) and residual distribution (right).

    A well-fitting linear model has residuals symmetrically around zero and
    no pattern in the actual-vs-predicted plot.
    """
    y_pred = pipe.predict(X_test)
    residuals = y_test - y_pred
    r2 = 1 - np.sum(residuals**2) / np.sum((y_test - y_test.mean())**2)

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[
            f"Actual vs. Predicted (R² = {r2:.3f})",
            "Residual Distribution",
        ],
        column_widths=[0.6, 0.4],
        horizontal_spacing=0.1,
    )

    # — Left: Actual vs Predicted ——————————————————————————————————
    resid_abs = np.abs(residuals)
    cmax = float(np.percentile(resid_abs, 95))
    fig.add_trace(go.Scatter(
        x=y_pred, y=y_test,
        mode="markers",
        marker=dict(
            color=resid_abs,
            colorscale="RdYlGn_r",
            cmin=0, cmax=cmax,
            size=5, opacity=0.6,
            showscale=False,   # colorbar removed to avoid subplot overlap
        ),
        name="Test instances",
        hovertemplate="Predicted: %{x:.0f}<br>Actual: %{y:.0f}<extra></extra>",
    ), row=1, col=1)

    # Inline color legend
    fig.add_annotation(
        row=1, col=1,
        xref="x domain", yref="y domain",
        x=0.03, y=0.97,
        text="● green = small error<br>● red = large error",
        showarrow=False, align="left", xanchor="left", yanchor="top",
        font=dict(size=11),
        bgcolor="rgba(255,255,255,0.85)",
        bordercolor="#D1D5DB", borderwidth=1,
    )

    lim = [min(y_pred.min(), y_test.min()), max(y_pred.max(), y_test.max())]
    fig.add_trace(go.Scatter(
        x=lim, y=lim, mode="lines",
        line=dict(dash="dash", color=COLORS["neutral"], width=1.5),
        name="Perfect fit",
        showlegend=True,
    ), row=1, col=1)

    fig.update_xaxes(title_text="Predicted rentals", row=1, col=1)
    fig.update_yaxes(title_text="Actual rentals", row=1, col=1)

    # — Right: Residual Distribution ———————————————————————————————
    # Histogram
    fig.add_trace(go.Histogram(
        x=residuals,
        nbinsx=40,
        marker_color=COLORS["primary"],
        opacity=0.7,
        name="Residuals",
        histnorm="probability density",
        hovertemplate="Residual: %{x:.0f}<br>Density: %{y:.4f}<extra></extra>",
    ), row=1, col=2)

    # Fitted normal curve
    mu, sigma = float(residuals.mean()), float(residuals.std())
    x_norm = np.linspace(residuals.min(), residuals.max(), 200)
    y_norm = stats.norm.pdf(x_norm, mu, sigma)
    fig.add_trace(go.Scatter(
        x=x_norm, y=y_norm, mode="lines",
        line=dict(color=COLORS["accent"], width=2),
        name="Normal fit",
    ), row=1, col=2)

    fig.add_vline(x=0, line_dash="dash", line_color=COLORS["neutral"],
                  line_width=1, row=1, col=2)

    fig.add_annotation(
        x=0.97, y=0.95, xref="paper", yref="paper",
        text=f"μ = {mu:+.0f}<br>σ = {sigma:.0f}",
        showarrow=False, align="right",
        font=dict(size=12),
        bgcolor="rgba(255,255,255,0.85)",
    )

    fig.update_xaxes(title_text="Residual (actual − predicted)", row=1, col=2)
    fig.update_yaxes(title_text="Density", row=1, col=2)

    fig.update_layout(
        title="Linear Regression — Diagnostics: Where Does It Fit Well?",
        height=500,
        legend=dict(orientation="h", y=-0.18),
        margin=dict(t=80),
    )
    return fig


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    X_train, X_test, y_train, y_test, feature_names = load_regression()
    pipe = fit_model(X_train, y_train)

    print("Figure 1: Feature effect and residuals…")
    fig1 = plot_lm_fit(pipe, X_train, y_train, feature_names)
    save_figure(fig1, CHAPTER, "how_lm_fit")

    print("Figure 2: Prediction contribution waterfall…")
    fig2 = plot_lm_contributions(pipe, X_test, y_test, feature_names)
    save_figure(fig2, CHAPTER, "how_lm_contributions")

    print("Figure 3: Diagnostics…")
    fig3 = plot_lm_diagnostics(pipe, X_test, y_test)
    save_figure(fig3, CHAPTER, "how_lm_diagnostics")


if __name__ == "__main__":
    main()

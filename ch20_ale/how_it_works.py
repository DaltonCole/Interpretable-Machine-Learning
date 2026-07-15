"""
Chapter 20: How ALE (Accumulated Local Effects) Plots Work
https://christophm.github.io/interpretable-ml-book/ale.html

Conceptual figures explaining ALE — separate from the model-output figures in main.py.

Figures produced:
  1. how_ale_vs_pdp_corr.html/png      — ALE vs PDP for correlated features (temp, atemp)
  2. how_ale_intervals.html/png        — step-by-step ALE computation for 'hr'
  3. how_ale_feature_comparison.html/png — ALE for all 12 features in a small-multiples grid
"""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from shared.datasets import load_regression
from shared.models import get_regression_model
from shared.theme import COLORS, save_figure

CHAPTER = "ch20_ale"
RNG = np.random.default_rng(42)


# ── Shared helpers ─────────────────────────────────────────────────────────────

def compute_pdp(model, X, feature_idx, grid):
    """Average prediction when feature_idx is fixed to each value in grid."""
    pdp = np.zeros(len(grid))
    for j, val in enumerate(grid):
        X_mod = X.copy()
        X_mod[:, feature_idx] = val
        pdp[j] = model.predict(X_mod).mean()
    return pdp


def compute_ale(model, X, feature_idx, n_bins=20):
    """Accumulated Local Effects via quantile-based bins.

    Returns (midpoints, ale_curve) where ale_curve is centred at zero.
    The local effect in each bin is the average finite-difference
    f(upper_bound) − f(lower_bound) for instances that fall in the bin.
    """
    feat = X[:, feature_idx]
    # Quantile-based bin edges (robust to discrete / skewed features)
    quantiles = np.percentile(feat, np.linspace(0, 100, n_bins + 1))
    quantiles = np.unique(quantiles)           # remove duplicate edges (discrete features)
    n_actual = len(quantiles) - 1
    if n_actual == 0:
        return np.array([feat.mean()]), np.array([0.0])

    midpoints = (quantiles[:-1] + quantiles[1:]) / 2.0
    local_effects = np.zeros(n_actual)

    for k in range(n_actual):
        lo, hi = quantiles[k], quantiles[k + 1]
        mask = (feat >= lo) & (feat <= hi)
        if mask.sum() == 0:
            continue
        X_lo = X[mask].copy();  X_lo[:, feature_idx] = lo
        X_hi = X[mask].copy();  X_hi[:, feature_idx] = hi
        local_effects[k] = (model.predict(X_hi) - model.predict(X_lo)).mean()

    ale_curve = np.cumsum(local_effects)
    ale_curve -= ale_curve.mean()   # centre at zero
    return midpoints, ale_curve


# ── Figure 1: ALE vs PDP for correlated features ──────────────────────────────

def plot_ale_vs_pdp_corr(model, X_train, feature_names) -> go.Figure:
    """Two panels:
      Left  — scatter of temp vs atemp showing r≈0.99 correlation; overlay the
               'extrapolation zone' where PDP sends predictions but data is sparse.
      Right — overlay PDP and ALE curves for 'temp' on the same axis.
    PDP averages over ALL atemp values for each fixed temp, including unrealistic
    (temp=0.1, atemp=0.9) combinations.  ALE restricts differences to local,
    realistic intervals and therefore avoids extrapolation.
    """
    temp_idx = feature_names.index("temp")
    atemp_idx = feature_names.index("atemp")

    temp_vals = X_train[:, temp_idx]
    atemp_vals = X_train[:, atemp_idx]
    r = float(np.corrcoef(temp_vals, atemp_vals)[0, 1])

    # Grids
    temp_grid = np.linspace(temp_vals.min(), temp_vals.max(), 30)
    X_sub = X_train[:500]

    pdp_temp = compute_pdp(model, X_sub, temp_idx, temp_grid)
    ale_mids, ale_temp = compute_ale(model, X_train, temp_idx, n_bins=20)

    # Scale ALE to prediction space (add mean prediction as offset for visual alignment)
    mean_pred = model.predict(X_sub).mean()
    ale_temp_abs = ale_temp + mean_pred

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[
            f"Correlation: temp vs atemp (r = {r:.3f})",
            "PDP vs ALE for 'temp'",
        ],
        column_widths=[0.48, 0.52],
        horizontal_spacing=0.12,
    )

    # ── Left: correlation scatter ──
    samp = RNG.choice(len(X_train), size=600, replace=False)
    fig.add_trace(go.Scatter(
        x=X_train[samp, temp_idx],
        y=X_train[samp, atemp_idx],
        mode="markers",
        marker=dict(color=COLORS["neutral"], size=4, opacity=0.35),
        name="Training data",
        hovertemplate="temp=%{x:.2f}<br>atemp=%{y:.2f}<extra></extra>",
    ), row=1, col=1)

    # Diagonal reference: temp = atemp (the realistic region)
    diag = np.linspace(min(temp_vals.min(), atemp_vals.min()),
                       max(temp_vals.max(), atemp_vals.max()), 2)
    fig.add_trace(go.Scatter(
        x=diag, y=diag,
        mode="lines",
        line=dict(color=COLORS["positive"], width=2, dash="dash"),
        name="temp = atemp (realistic)",
        hoverinfo="skip",
    ), row=1, col=1)

    # Shade unrealistic regions: where |temp - atemp| > 0.15
    t_lo = float(temp_vals.min())
    t_hi = float(temp_vals.max())
    gap = 0.15

    # Upper unrealistic region (atemp >> temp)
    fig.add_shape(
        type="rect",
        x0=t_lo, x1=t_hi,
        y0=t_lo + gap, y1=float(atemp_vals.max()) + 0.05,
        fillcolor="rgba(220,38,38,0.08)",
        line=dict(width=0),
        row=1, col=1,
    )
    # Lower unrealistic region (temp >> atemp)
    fig.add_shape(
        type="rect",
        x0=t_lo, x1=t_hi,
        y0=float(atemp_vals.min()) - 0.05, y1=t_lo - gap,
        fillcolor="rgba(220,38,38,0.08)",
        line=dict(width=0),
        row=1, col=1,
    )

    fig.add_annotation(
        row=1, col=1,
        xref="x domain", yref="y domain",
        x=0.03, y=0.98,
        text="<span style='color:#DC2626'>■ PDP extrapolates here</span><br>(rare in real data)",
        showarrow=False, align="left", font=dict(size=10),
        bgcolor="rgba(255,255,255,0.85)", bordercolor=COLORS["negative"], borderwidth=1,
        xanchor="left", yanchor="top",
    )

    fig.update_xaxes(title_text="temp (normalised temperature)", row=1, col=1)
    fig.update_yaxes(title_text="atemp (feels-like temperature)", row=1, col=1)

    # ── Right: PDP vs ALE overlay ──
    fig.add_trace(go.Scatter(
        x=temp_grid, y=pdp_temp,
        mode="lines",
        line=dict(color=COLORS["accent"], width=2.5),
        name="PDP(temp) — may extrapolate",
        hovertemplate="temp=%{x:.2f}<br>PDP=%{y:.0f}<extra></extra>",
    ), row=1, col=2)

    fig.add_trace(go.Scatter(
        x=ale_mids, y=ale_temp_abs,
        mode="lines",
        line=dict(color=COLORS["primary"], width=2.5),
        name="ALE(temp) — stays realistic",
        hovertemplate="temp=%{x:.2f}<br>ALE=%{y:.0f}<extra></extra>",
    ), row=1, col=2)

    fig.add_annotation(
        row=1, col=2,
        xref="x2 domain", yref="y2 domain",
        x=0.03, y=0.98,
        text=(
            "<b>ALE</b> (blue) uses local finite differences<br>"
            "within realistic data intervals<br>"
            "<b>PDP</b> (pink) averages globally,<br>"
            "including unrealistic (temp, atemp) pairs"
        ),
        showarrow=False, align="left", font=dict(size=10),
        bgcolor="rgba(255,255,255,0.9)", bordercolor=COLORS["neutral"], borderwidth=1,
        xanchor="left", yanchor="top",
    )

    fig.update_xaxes(title_text="temp (normalised temperature)", row=1, col=2)
    fig.update_yaxes(title_text="Predicted bike rentals", row=1, col=2)

    fig.update_layout(
        title=(
            "ALE vs PDP — When Features Are Correlated, PDP Uses Unrealistic Combinations<br>"
            "<sup>temp and atemp are highly correlated (r ≈ 0.99) — PDP extrapolates, ALE does not</sup>"
        ),
        legend=dict(orientation="h", y=-0.18),
        height=520,
        margin=dict(t=90, b=80),
    )
    return fig


# ── Figure 2: ALE interval step-by-step for 'hr' ─────────────────────────────

def plot_ale_intervals(model, X_train, feature_names) -> go.Figure:
    """Two stacked panels showing the ALE computation for 'hr' step by step.

    Top:    local effect (bar) within each interval — the raw finite difference.
    Bottom: accumulated local effects (line) — the actual ALE curve.
    """
    hr_idx = feature_names.index("hr")
    feat = X_train[:, hr_idx]

    n_bins = 10
    quantiles = np.percentile(feat, np.linspace(0, 100, n_bins + 1))
    quantiles = np.unique(quantiles)
    n_actual = len(quantiles) - 1

    midpoints = (quantiles[:-1] + quantiles[1:]) / 2.0
    local_effects = np.zeros(n_actual)
    bin_counts = np.zeros(n_actual, dtype=int)

    for k in range(n_actual):
        lo, hi = quantiles[k], quantiles[k + 1]
        mask = (feat >= lo) & (feat <= hi)
        bin_counts[k] = mask.sum()
        if mask.sum() == 0:
            continue
        X_lo = X_train[mask].copy();  X_lo[:, hr_idx] = lo
        X_hi = X_train[mask].copy();  X_hi[:, hr_idx] = hi
        local_effects[k] = (model.predict(X_hi) - model.predict(X_lo)).mean()

    ale_curve = np.cumsum(local_effects)
    ale_curve -= ale_curve.mean()

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        subplot_titles=[
            "Step 1 — Local Effect per Interval (finite difference within each bin)",
            "Step 2 — Accumulated Effects → ALE Curve (centred at zero)",
        ],
        vertical_spacing=0.14,
        row_heights=[0.5, 0.5],
    )

    # Top: local effects bar chart
    bar_colors = [COLORS["positive"] if v >= 0 else COLORS["negative"] for v in local_effects]
    fig.add_trace(go.Bar(
        x=midpoints,
        y=local_effects,
        marker_color=bar_colors,
        name="Local effect",
        width=(midpoints[1] - midpoints[0]) * 0.8 if len(midpoints) > 1 else 1.0,
        hovertemplate=(
            "Interval midpoint: %{x:.1f}<br>"
            "Local Δ: %{y:+.1f} rentals<extra></extra>"
        ),
        texttemplate="%{y:+.0f}",
        textposition="outside",
    ), row=1, col=1)
    fig.add_hline(y=0, line_dash="dot", line_color=COLORS["neutral"], line_width=1, row=1, col=1)
    fig.update_yaxes(title_text="Local Δ prediction (rentals)", row=1, col=1)

    # Rug: data distribution of hr
    samp = RNG.choice(len(X_train), size=500, replace=False)
    rug_y = float(local_effects.min()) - abs(float(local_effects.min())) * 0.12 - 5
    fig.add_trace(go.Scatter(
        x=X_train[samp, hr_idx],
        y=np.full(len(samp), rug_y),
        mode="markers",
        marker=dict(
            symbol="line-ns",
            size=5,
            color=COLORS["neutral"],
            opacity=0.3,
            line=dict(color=COLORS["neutral"], width=1),
        ),
        name="Data density (rug)",
        showlegend=True,
        hoverinfo="skip",
    ), row=1, col=1)

    # Bottom: accumulated ALE curve
    fig.add_trace(go.Scatter(
        x=midpoints, y=ale_curve,
        mode="lines+markers",
        line=dict(color=COLORS["primary"], width=3),
        marker=dict(size=8, color=COLORS["primary"]),
        name="ALE(hr)",
        hovertemplate="hr=%{x:.1f}<br>ALE=%{y:+.1f} rentals<extra></extra>",
        fill="tozeroy",
        fillcolor="rgba(37,99,235,0.08)",
    ), row=2, col=1)
    fig.add_hline(y=0, line_dash="dot", line_color=COLORS["neutral"], line_width=1, row=2, col=1)
    fig.update_xaxes(title_text="Hour of day (hr)", row=2, col=1)
    fig.update_yaxes(title_text="Accumulated effect (rentals)", row=2, col=1)

    fig.update_layout(
        title=(
            "ALE — Local Differences Within Intervals Avoid Extrapolation<br>"
            "<sup>Each bar = average Δprediction at interval boundaries "
            "| accumulating gives the ALE curve</sup>"
        ),
        legend=dict(orientation="h", y=-0.12),
        height=520,
        margin=dict(t=90, b=60),
    )
    return fig


# ── Figure 3: ALE small-multiples for all 12 features ─────────────────────────

def plot_ale_feature_comparison(model, X_train, feature_names) -> go.Figure:
    """3×4 grid of ALE curves for every feature, all centred at zero.

    The subplot with the widest y-range (expected: 'hr') corresponds to the
    most influential feature — a direct visual feature ranking.
    """
    n_features = len(feature_names)   # 12 for bike dataset
    rows, cols = 3, 4

    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=feature_names,
        vertical_spacing=0.13,
        horizontal_spacing=0.08,
    )

    for i, fname in enumerate(feature_names):
        row = i // cols + 1
        col = i % cols + 1

        mids, ale = compute_ale(model, X_train, i, n_bins=20)

        fig.add_trace(go.Scatter(
            x=mids, y=ale,
            mode="lines",
            line=dict(color=COLORS["primary"], width=2),
            showlegend=False,
            hovertemplate=f"{fname}=%{{x:.2f}}<br>ALE=%{{y:+.1f}}<extra></extra>",
            fill="tozeroy",
            fillcolor="rgba(37,99,235,0.07)",
        ), row=row, col=col)

        fig.add_hline(y=0, line_dash="dot", line_color=COLORS["neutral"],
                      line_width=1, row=row, col=col)

    # Y-axis labels only for left-column subplots
    for r in range(1, rows + 1):
        fig.update_yaxes(title_text="ALE", title_font=dict(size=10), row=r, col=1)

    fig.update_layout(
        title=(
            "ALE Plots — Compare All Features' Marginal Effects (Centred at Zero)<br>"
            "<sup>Subplot y-range reflects effect magnitude — wider range = stronger feature | "
            "all curves centred at zero for fair comparison</sup>"
        ),
        height=600,
        margin=dict(t=90, b=40),
    )
    return fig


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    X_train, X_test, y_train, y_test, feature_names = load_regression()
    model = get_regression_model(X_train, y_train)

    print("Figure 1: ALE vs PDP for correlated features (temp/atemp)…")
    fig1 = plot_ale_vs_pdp_corr(model, X_train, feature_names)
    save_figure(fig1, CHAPTER, "how_ale_vs_pdp_corr")

    print("Figure 2: ALE interval step-by-step for 'hr'…")
    fig2 = plot_ale_intervals(model, X_train, feature_names)
    save_figure(fig2, CHAPTER, "how_ale_intervals")

    print("Figure 3: ALE small-multiples for all 12 features…")
    fig3 = plot_ale_feature_comparison(model, X_train, feature_names)
    save_figure(fig3, CHAPTER, "how_ale_feature_comparison")


if __name__ == "__main__":
    main()

"""
Chapter 21: How Feature Interaction Detection Works
https://christophm.github.io/interpretable-ml-book/interaction.html

Conceptual figures explaining feature interactions and the H-statistic —
separate from the model-output figures in main.py.

Figures produced:
  1. how_interaction_concept.html/png  — synthetic heatmaps: with vs without interaction
  2. how_hstat_concept.html/png        — 2D PDP vs additive decomposition (H-statistic)
  3. how_interaction_season_hr.html/png — PDP of hr per season (real interaction)
"""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from shared.datasets import load_regression
from shared.models import get_regression_model
from shared.theme import COLORS, save_figure

CHAPTER = "ch21_feature_interaction"
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


# ── Figure 1: Synthetic interaction concept ───────────────────────────────────

def plot_interaction_concept() -> go.Figure:
    """Two heatmaps on a synthetic 2-feature space:
      Left:  f(x1, x2) = x1 + x2 + x1·x2  (with interaction)
      Right: f(x1, x2) = x1 + x2             (purely additive)

    The interaction term tilts the heatmap diagonally — equal-value contours are
    no longer straight horizontal or vertical lines.
    """
    x1 = np.linspace(-1, 1, 40)
    x2 = np.linspace(-1, 1, 40)
    X1, X2 = np.meshgrid(x1, x2)

    z_interact = X1 + X2 + X1 * X2       # with interaction term
    z_additive = X1 + X2                  # purely additive

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[
            "f(x1, x2) = x1 + x2 + x1·x2  (interaction)",
            "f(x1, x2) = x1 + x2  (additive — no interaction)",
        ],
        horizontal_spacing=0.10,
    )

    shared_colorscale = "RdBu_r"
    z_min = min(z_interact.min(), z_additive.min())
    z_max = max(z_interact.max(), z_additive.max())

    # Left: with interaction
    fig.add_trace(go.Heatmap(
        x=x1, y=x2, z=z_interact,
        colorscale=shared_colorscale,
        zmin=z_min, zmax=z_max,
        showscale=False,
        hovertemplate="x1=%{x:.2f}<br>x2=%{y:.2f}<br>f=%{z:.2f}<extra></extra>",
        name="With interaction",
    ), row=1, col=1)

    # Right: additive (no interaction) — shared colour scale
    fig.add_trace(go.Heatmap(
        x=x1, y=x2, z=z_additive,
        colorscale=shared_colorscale,
        zmin=z_min, zmax=z_max,
        colorbar=dict(title="f(x1,x2)", len=0.8, y=0.5),
        hovertemplate="x1=%{x:.2f}<br>x2=%{y:.2f}<br>f=%{z:.2f}<extra></extra>",
        name="Additive",
    ), row=1, col=2)

    # x/y axis labels
    for col in (1, 2):
        fig.update_xaxes(title_text="x1", row=1, col=col)
        fig.update_yaxes(title_text="x2", row=1, col=col)

    # Annotation pointing out the diagonal tilt in the interaction panel
    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.23, y=0.02,
        text=(
            "Diagonal tilt = interaction<br>"
            "The effect of x1 depends on x2"
        ),
        showarrow=False, align="center",
        font=dict(size=11, color=COLORS["negative"]),
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor=COLORS["negative"], borderwidth=1,
    )
    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.73, y=0.02,
        text=(
            "Parallel stripes = no interaction<br>"
            "Effects are independent and add up"
        ),
        showarrow=False, align="center",
        font=dict(size=11, color=COLORS["positive"]),
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor=COLORS["positive"], borderwidth=1,
    )

    fig.update_layout(
        title=(
            "Feature Interaction — When the Joint Effect Exceeds the Sum of Individual Effects<br>"
            "<sup>Interaction = the effect of x1 on f depends on the value of x2 (and vice versa)</sup>"
        ),
        height=480,
        margin=dict(t=90, b=70, r=80),
    )
    return fig


# ── Figure 2: H-statistic via 2D PDP decomposition ───────────────────────────

def plot_hstat_concept(model, X_train, feature_names) -> go.Figure:
    """Left:  actual 2D PDP of (hr, season).
    Right: expected additive surface = pdp_hr + pdp_season (no-interaction baseline).

    The H-statistic measures how much the actual 2D PDP deviates from the additive
    baseline — distance from left to right panel.
    """
    hr_idx = feature_names.index("hr")
    season_idx = feature_names.index("season")

    hr_grid = np.arange(0, 24)
    season_grid = np.array([1.0, 2.0, 3.0, 4.0])

    X_pdp = X_train[:300]   # subsample for speed

    print("  Computing 2D PDP (hr × season)…")
    pdp2d = np.zeros((len(hr_grid), len(season_grid)))
    for i, hv in enumerate(hr_grid):
        for j, sv in enumerate(season_grid):
            X_mod = X_pdp.copy()
            X_mod[:, hr_idx] = hv
            X_mod[:, season_idx] = sv
            pdp2d[i, j] = model.predict(X_mod).mean()

    pdp_hr = pdp2d.mean(axis=1)        # 1D PDP of hr (marginalise over season)
    pdp_season = pdp2d.mean(axis=0)    # 1D PDP of season (marginalise over hr)

    # Centred versions for H-statistic
    pdp2d_c = pdp2d - pdp2d.mean()
    pdp_hr_c = pdp_hr - pdp_hr.mean()
    pdp_season_c = pdp_season - pdp_season.mean()

    additive = pdp_hr_c[:, None] + pdp_season_c[None, :]   # shape (24, 4)

    # H-statistic
    denom = np.sum(pdp2d_c ** 2)
    h_sq = float(np.sum((pdp2d_c - additive) ** 2) / denom) if denom > 0 else 0.0
    h_val = float(np.sqrt(max(0.0, h_sq)))

    # Re-add mean to additive for display (same absolute scale as pdp2d)
    additive_abs = additive + pdp2d.mean()

    season_labels = ["Spring", "Summer", "Fall", "Winter"]
    shared_colorscale = "RdBu_r"
    z_min = min(pdp2d.min(), additive_abs.min())
    z_max = max(pdp2d.max(), additive_abs.max())

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[
            "Actual 2D PDP: hr × season",
            "Expected Additive: PDP(hr) + PDP(season)",
        ],
        horizontal_spacing=0.12,
    )

    fig.add_trace(go.Heatmap(
        x=season_labels,
        y=hr_grid,
        z=pdp2d,
        colorscale=shared_colorscale,
        zmin=z_min, zmax=z_max,
        showscale=False,
        hovertemplate="season=%{x}<br>hr=%{y}<br>Avg prediction=%{z:.0f}<extra></extra>",
    ), row=1, col=1)

    fig.add_trace(go.Heatmap(
        x=season_labels,
        y=hr_grid,
        z=additive_abs,
        colorscale=shared_colorscale,
        zmin=z_min, zmax=z_max,
        colorbar=dict(title="Avg<br>rentals", len=0.8, y=0.5),
        hovertemplate="season=%{x}<br>hr=%{y}<br>Additive=%{z:.0f}<extra></extra>",
    ), row=1, col=2)

    for col in (1, 2):
        fig.update_xaxes(title_text="Season", row=1, col=col)
        fig.update_yaxes(title_text="Hour of day (hr)", row=1, col=col)

    # H-statistic annotation
    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.5, y=1.01,
        text=(
            f"H = {h_val:.3f}   (H² = {h_sq * 100:.1f}% of joint variance due to interaction) "
            "| H=0 means no interaction, H=1 means full interaction"
        ),
        showarrow=False, align="center",
        font=dict(size=11, color=COLORS["primary"]),
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor=COLORS["primary"], borderwidth=1,
    )
    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.5, y=-0.08,
        text=(
            "H-statistic = distance between left and right panels | "
            "if they look the same, there is no interaction"
        ),
        showarrow=False, align="center",
        font=dict(size=11, color=COLORS["neutral"]),
    )

    fig.update_layout(
        title=(
            "H-Statistic — Measuring Interaction Strength via PDP Decomposition<br>"
            "<sup>Left = actual 2D PDP | Right = what it would look like with no hr×season interaction</sup>"
        ),
        height=500,
        margin=dict(t=110, b=60, r=80),
    )
    return fig


# ── Figure 3: hr PDP per season (real interaction) ───────────────────────────

def plot_interaction_season_hr(model, X_train, feature_names) -> go.Figure:
    """Four PDP curves for 'hr' — one per season.

    If the curves have different shapes (and they will), it confirms that hr and
    season interact: the effect of hour on bike rentals changes with the season.
    """
    hr_idx = feature_names.index("hr")
    season_idx = feature_names.index("season")
    hr_grid = np.arange(0, 24)

    season_info = [
        (1, "Spring"),
        (2, "Summer"),
        (3, "Fall"),
        (4, "Winter"),
    ]

    fig = go.Figure()

    season_pdps = {}
    for season_val, season_name in season_info:
        mask = X_train[:, season_idx] == season_val
        X_sub = X_train[mask]
        if len(X_sub) == 0:
            continue
        pdp = np.zeros(len(hr_grid))
        for j, hv in enumerate(hr_grid):
            X_mod = X_sub.copy()
            X_mod[:, hr_idx] = hv
            pdp[j] = model.predict(X_mod).mean()
        season_pdps[season_val] = pdp

        color_idx = season_val - 1
        fig.add_trace(go.Scatter(
            x=hr_grid, y=pdp,
            mode="lines",
            line=dict(color=COLORS["palette"][color_idx], width=2.5),
            name=season_name,
            hovertemplate=f"{season_name}: hr=%{{x}}<br>Avg prediction=%{{y:.0f}}<extra></extra>",
        ))

    # Annotate the peak difference at hr=17 (evening rush)
    if 2 in season_pdps and 4 in season_pdps:   # Summer vs Winter
        summer_peak = float(season_pdps[2][17])
        winter_peak = float(season_pdps[4][17])
        diff = summer_peak - winter_peak

        fig.add_annotation(
            x=17, y=summer_peak + 15,
            text=f"Summer peak: {summer_peak:.0f}<br>Winter peak: {winter_peak:.0f}<br>Δ = {diff:.0f} rentals",
            showarrow=True, arrowhead=2, ax=50, ay=-50,
            font=dict(size=11, color=COLORS["accent"]),
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor=COLORS["accent"], borderwidth=1,
        )

    # Mark the morning rush as well
    fig.add_vline(x=8, line_dash="dot", line_color=COLORS["neutral"], line_width=1,
                  annotation_text="8 AM rush", annotation_position="top right",
                  annotation_font=dict(size=10, color=COLORS["neutral"]))
    fig.add_vline(x=17, line_dash="dot", line_color=COLORS["neutral"], line_width=1,
                  annotation_text="5 PM rush", annotation_position="top left",
                  annotation_font=dict(size=10, color=COLORS["neutral"]))

    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.02, y=0.97,
        text=(
            "If there were <b>no interaction</b>, all 4 curves<br>"
            "would have the same shape (just shifted up/down).<br>"
            "Different shapes confirm hr × season interaction."
        ),
        showarrow=False, align="left",
        font=dict(size=11),
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor=COLORS["primary"], borderwidth=1,
        xanchor="left", yanchor="top",
    )

    fig.update_layout(
        title=(
            "Feature Interaction — Hour's Effect on Rentals Changes by Season<br>"
            "<sup>PDP of hr computed separately within each season | "
            "different curve shapes = confirmed interaction</sup>"
        ),
        xaxis=dict(title="Hour of day (hr)", tickmode="linear", tick0=0, dtick=2),
        yaxis_title="Average predicted bike rentals",
        legend=dict(
            title="Season",
            orientation="h", y=-0.18,
        ),
        height=500,
        margin=dict(t=90, b=70),
    )
    return fig


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    X_train, X_test, y_train, y_test, feature_names = load_regression()
    model = get_regression_model(X_train, y_train)

    print("Figure 1: Synthetic interaction concept…")
    fig1 = plot_interaction_concept()
    save_figure(fig1, CHAPTER, "how_interaction_concept")

    print("Figure 2: H-statistic via 2D PDP decomposition (hr × season)…")
    fig2 = plot_hstat_concept(model, X_train, feature_names)
    save_figure(fig2, CHAPTER, "how_hstat_concept")

    print("Figure 3: PDP of hr per season (real interaction)…")
    fig3 = plot_interaction_season_hr(model, X_train, feature_names)
    save_figure(fig3, CHAPTER, "how_interaction_season_hr")


if __name__ == "__main__":
    main()

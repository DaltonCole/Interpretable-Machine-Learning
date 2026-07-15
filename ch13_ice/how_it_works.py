"""
Chapter 13: How ICE Plots Work
https://christophm.github.io/interpretable-ml-book/ice.html

Conceptual figures explaining Individual Conditional Expectation plots.

Figures produced:
  1. how_ice_concept.html/png      — 150 ICE lines + PDP + std band for hr
  2. how_cice_concept.html/png     — centred ICE highlighting two extreme subgroups
  3. how_ice_heterogeneity.html/png — left: ICE; right: per-hour std bar chart
"""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from shared.datasets import load_regression
from shared.models import get_regression_model
from shared.theme import COLORS, save_figure

CHAPTER = "ch13_ice"
RNG = np.random.default_rng(42)
N_ICE = 150      # instances to draw
N_POINTS = 24    # one grid point per integer hour 0–23


# ── Helper ────────────────────────────────────────────────────────────────────

def compute_ice_matrix(model, X, feature_idx, n_points=N_POINTS):
    """Return (grid, ice_matrix) — ice_matrix shape (n_instances, n_points)."""
    grid = np.linspace(X[:, feature_idx].min(), X[:, feature_idx].max(), n_points)
    ice = np.zeros((len(X), n_points))
    for j, val in enumerate(grid):
        X_mod = X.copy()
        X_mod[:, feature_idx] = val
        ice[:, j] = model.predict(X_mod)
    return grid, ice


# ── Figure 1: ICE + PDP + std band ───────────────────────────────────────────

def plot_ice_concept(model, X_sample, feature_names) -> go.Figure:
    """150 ICE lines for hr, PDP overlay, ±1 std shaded band."""
    hr_idx = feature_names.index("hr")
    grid, ice = compute_ice_matrix(model, X_sample, hr_idx)
    pdp = ice.mean(axis=0)
    std = ice.std(axis=0)

    fig = go.Figure()

    # ±1 std band around PDP
    fig.add_trace(go.Scatter(
        x=np.concatenate([grid, grid[::-1]]),
        y=np.concatenate([pdp + std, (pdp - std)[::-1]]),
        fill="toself",
        fillcolor="rgba(219,39,119,0.12)",
        line=dict(color="rgba(0,0,0,0)"),
        name="PDP ± 1 std",
        hoverinfo="skip",
    ))

    # Individual ICE lines
    for i in range(len(ice)):
        fig.add_trace(go.Scatter(
            x=grid, y=ice[i],
            mode="lines",
            line=dict(color="rgba(37,99,235,0.14)", width=1),
            showlegend=False,
            hoverinfo="skip",
        ))

    # PDP (mean)
    fig.add_trace(go.Scatter(
        x=grid, y=pdp,
        mode="lines",
        line=dict(color=COLORS["accent"], width=3.5),
        name="PDP (average over all lines)",
        hovertemplate="hr=%{x:.0f}<br>PDP=%{y:.0f}<extra></extra>",
    ))

    # Annotations pointing to a single ICE line and the PDP
    fig.add_annotation(
        x=float(grid[8]), y=float(ice[5, 8]) + 60,
        text="Each line = one instance",
        showarrow=True, arrowhead=2, ax=40, ay=-30,
        font=dict(size=12, color=COLORS["primary"]),
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor=COLORS["primary"], borderwidth=1,
    )
    fig.add_annotation(
        x=float(grid[17]), y=float(pdp[17]) + 70,
        text="Bold line = average (PDP)",
        showarrow=True, arrowhead=2, ax=0, ay=-38,
        font=dict(size=12, color=COLORS["accent"]),
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor=COLORS["accent"], borderwidth=1,
    )

    fig.update_layout(
        title=("ICE Plot — One Prediction Curve Per Instance<br>"
               "<sup>Each thin blue line is one instance's prediction as 'hr' varies 0–23. "
               "The bold pink line is the average (PDP). Shading = ±1 std.</sup>"),
        xaxis=dict(title="Hour of day", tickmode="linear", tick0=0, dtick=4),
        yaxis_title="Predicted bike rentals",
        height=520,
        legend=dict(orientation="h", y=-0.18),
    )
    return fig


# ── Figure 2: centred ICE with subgroup highlighting ──────────────────────────

def plot_cice_concept(model, X_sample, feature_names) -> go.Figure:
    """Centred ICE — subtract prediction at hr=0 so all lines start at 0.

    Highlights two extreme instances: one with a strong morning-rush peak and
    one that rises monotonically throughout the day.
    """
    hr_idx = feature_names.index("hr")
    season_idx = feature_names.index("season")
    weekday_idx = feature_names.index("weekday")

    grid, ice = compute_ice_matrix(model, X_sample, hr_idx)
    cice = ice - ice[:, [0]]  # subtract prediction at hr=0
    cpdp = cice.mean(axis=0)

    # Identify extreme instances
    # Morning rush: high peak at hr≈8–9, then drops off by hr≈14
    peak_am = cice[:, 8:10].max(axis=1)
    val_pm = cice[:, 14].copy()
    rush_score = peak_am - val_pm
    morning_rush_idx = int(np.argmax(rush_score))

    # Monotonic rise: high Pearson correlation of cice with grid
    corr = np.array([np.corrcoef(grid, cice[i])[0, 1] for i in range(len(cice))])
    monotonic_idx = int(np.argmax(corr))
    if monotonic_idx == morning_rush_idx:
        sorted_corr = np.argsort(corr)[::-1]
        monotonic_idx = int(sorted_corr[1])

    season_names = {1: "Spring", 2: "Summer", 3: "Fall", 4: "Winter"}
    weekday_names = {0: "Sun", 1: "Mon", 2: "Tue", 3: "Wed", 4: "Thu", 5: "Fri", 6: "Sat"}

    rush_inst = X_sample[morning_rush_idx]
    rush_season = season_names.get(int(rush_inst[season_idx]), "?")
    rush_wday = weekday_names.get(int(rush_inst[weekday_idx]), "?")

    mono_inst = X_sample[monotonic_idx]
    mono_season = season_names.get(int(mono_inst[season_idx]), "?")
    mono_wday = weekday_names.get(int(mono_inst[weekday_idx]), "?")

    fig = go.Figure()

    # Baseline at 0
    fig.add_hline(y=0, line_dash="dot", line_color=COLORS["neutral"], line_width=1)

    # All centred ICE lines
    for i in range(len(cice)):
        if i in (morning_rush_idx, monotonic_idx):
            continue
        fig.add_trace(go.Scatter(
            x=grid, y=cice[i],
            mode="lines",
            line=dict(color="rgba(107,114,128,0.15)", width=1),
            showlegend=False,
            hoverinfo="skip",
        ))

    # Centred PDP
    fig.add_trace(go.Scatter(
        x=grid, y=cpdp,
        mode="lines",
        line=dict(color=COLORS["neutral"], width=2.5, dash="dash"),
        name="c-PDP (centred average)",
    ))

    # Highlighted: morning rush
    fig.add_trace(go.Scatter(
        x=grid, y=cice[morning_rush_idx],
        mode="lines",
        line=dict(color=COLORS["accent"], width=3),
        name=f"Morning rush ({rush_season}, {rush_wday})",
    ))

    # Highlighted: monotonic rise
    fig.add_trace(go.Scatter(
        x=grid, y=cice[monotonic_idx],
        mode="lines",
        line=dict(color=COLORS["positive"], width=3),
        name=f"Monotonic rise ({mono_season}, {mono_wday})",
    ))

    # Annotations explaining the two instances
    peak_hr_idx = int(np.argmax(cice[morning_rush_idx]))
    fig.add_annotation(
        x=float(grid[peak_hr_idx]),
        y=float(cice[morning_rush_idx, peak_hr_idx]) + 30,
        text=f"Strong AM peak then drops<br>({rush_season}, {rush_wday})",
        showarrow=True, arrowhead=2, ax=40, ay=-30,
        font=dict(size=11, color=COLORS["accent"]),
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor=COLORS["accent"], borderwidth=1,
    )
    fig.add_annotation(
        x=float(grid[-3]),
        y=float(cice[monotonic_idx, -3]) + 30,
        text=f"Rises all day<br>({mono_season}, {mono_wday})",
        showarrow=True, arrowhead=2, ax=-50, ay=-30,
        font=dict(size=11, color=COLORS["positive"]),
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor=COLORS["positive"], borderwidth=1,
    )

    fig.update_layout(
        title=("Centred ICE — Subtract the Starting Point to Focus on Shape<br>"
               "<sup>All lines start at 0 (prediction at hr=0). "
               "Shape differences reveal fundamentally different usage patterns.</sup>"),
        xaxis=dict(title="Hour of day", tickmode="linear", tick0=0, dtick=4),
        yaxis_title="Δ prediction relative to hr=0",
        height=500,
        legend=dict(orientation="h", y=-0.18),
    )
    return fig


# ── Figure 3: ICE + per-hour disagreement ────────────────────────────────────

def plot_ice_heterogeneity(model, X_sample, feature_names) -> go.Figure:
    """Left: ICE plot (50 lines for clarity).
    Right: bar chart of prediction std at each hour — peaks mark disagreement.
    Top-3 disagreement hours are shaded on the left panel.
    """
    hr_idx = feature_names.index("hr")
    grid, ice = compute_ice_matrix(model, X_sample, hr_idx)
    std_per_hour = ice.std(axis=0)
    top3_hours = np.argsort(std_per_hour)[-3:]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[
            "ICE Lines for 'hr' (sample of 50)",
            "Prediction Std per Hour — Where Do Instances Disagree?",
        ],
        horizontal_spacing=0.12,
        column_widths=[0.55, 0.45],
    )

    # ── left: ICE (50 lines) ──
    draw_idx = RNG.choice(len(ice), size=50, replace=False)
    for i in draw_idx:
        fig.add_trace(go.Scatter(
            x=grid, y=ice[i],
            mode="lines",
            line=dict(color="rgba(37,99,235,0.18)", width=1),
            showlegend=False,
            hoverinfo="skip",
        ), row=1, col=1)

    # PDP overlay
    fig.add_trace(go.Scatter(
        x=grid, y=ice.mean(axis=0),
        mode="lines",
        line=dict(color=COLORS["accent"], width=2.5),
        name="PDP (mean)",
    ), row=1, col=1)

    # Shade top-3 disagreement hours on left panel
    y_lo = float(ice.min()) - 30
    y_hi = float(ice.max()) + 30
    for h_val in top3_hours:
        fig.add_shape(
            type="rect",
            x0=float(grid[h_val]) - 0.5,
            x1=float(grid[h_val]) + 0.5,
            y0=y_lo, y1=y_hi,
            fillcolor="rgba(220,38,38,0.13)",
            line_width=0,
            row=1, col=1,
        )

    # dummy trace for legend
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers",
        marker=dict(color="rgba(220,38,38,0.3)", size=12, symbol="square"),
        name="Top-3 disagreement hours",
    ))

    # ── right: std bar chart ──
    bar_colors = [
        COLORS["negative"] if i in top3_hours else COLORS["primary"]
        for i in range(len(grid))
    ]
    fig.add_trace(go.Bar(
        x=grid, y=std_per_hour,
        marker_color=bar_colors,
        name="Std of predictions",
        hovertemplate="hr=%{x:.0f}<br>Std=%{y:.0f}<extra></extra>",
    ), row=1, col=2)

    fig.add_annotation(
        x=0.98, y=0.97,
        xref="paper", yref="paper",
        text="Red bars = highest disagreement<br>between instances",
        showarrow=False,
        align="right", xanchor="right", yanchor="top",
        font=dict(size=11),
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor=COLORS["negative"], borderwidth=1,
    )

    fig.update_xaxes(title_text="Hour of day", tickmode="linear",
                     tick0=0, dtick=4, row=1, col=1)
    fig.update_yaxes(title_text="Predicted rentals", row=1, col=1)
    fig.update_xaxes(title_text="Hour of day", tickmode="linear",
                     tick0=0, dtick=4, row=1, col=2)
    fig.update_yaxes(title_text="Std of predictions", row=1, col=2)

    fig.update_layout(
        title=("ICE — Finding Heterogeneous Subgroups<br>"
               "<sup>Where the curves fan out widely, instances respond very differently "
               "to that hour — revealing hidden subgroups.</sup>"),
        height=490,
        legend=dict(orientation="h", y=-0.18),
    )
    return fig


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    X_train, X_test, y_train, y_test, feature_names = load_regression()
    model = get_regression_model(X_train, y_train)

    hr_idx = int(np.argmax(model.feature_importances_))
    print(f"Most important feature: {feature_names[hr_idx]}")

    # Sample 150 instances from X_train for ICE curves
    sample_idx = RNG.choice(len(X_train), size=N_ICE, replace=False)
    X_sample = X_train[sample_idx]

    print("Figure 1: ICE concept…")
    fig1 = plot_ice_concept(model, X_sample, feature_names)
    save_figure(fig1, CHAPTER, "how_ice_concept")

    print("Figure 2: Centred ICE…")
    fig2 = plot_cice_concept(model, X_sample, feature_names)
    save_figure(fig2, CHAPTER, "how_cice_concept")

    print("Figure 3: ICE heterogeneity…")
    fig3 = plot_ice_heterogeneity(model, X_sample, feature_names)
    save_figure(fig3, CHAPTER, "how_ice_heterogeneity")


if __name__ == "__main__":
    main()

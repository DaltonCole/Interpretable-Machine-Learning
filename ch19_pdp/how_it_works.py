"""
Chapter 19: How Partial Dependence Plots Work
https://christophm.github.io/interpretable-ml-book/pdp.html

Conceptual figures explaining the PDP method — separate from the model-output
figures in main.py.

Figures produced:
  1. how_pdp_concept.html/png    — ICE curves + their average = PDP (hr feature)
  2. how_pdp_comparison.html/png — 2×2 PDP grid for hr, temp, hum, season
  3. how_pdp_2d_concept.html/png — 2D PDP heatmap (hr × temp) with 1D PDPs below
"""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from shared.datasets import load_regression
from shared.models import get_regression_model
from shared.theme import COLORS, save_figure

CHAPTER = "ch19_pdp"
RNG = np.random.default_rng(42)


# ── Shared helpers ─────────────────────────────────────────────────────────────

def compute_pdp(model, X, feature_idx, grid):
    """Average prediction when feature_idx is set to each value in grid."""
    pdp = np.zeros(len(grid))
    for j, val in enumerate(grid):
        X_mod = X.copy()
        X_mod[:, feature_idx] = val
        pdp[j] = model.predict(X_mod).mean()
    return pdp


def compute_ice(model, X, feature_idx, grid):
    """ICE curves: (n_instances, len(grid)) predictions for each grid value."""
    ice = np.zeros((len(X), len(grid)))
    for j, val in enumerate(grid):
        X_mod = X.copy()
        X_mod[:, feature_idx] = val
        ice[:, j] = model.predict(X_mod)
    return ice


# ── Figure 1: ICE + PDP concept ───────────────────────────────────────────────

def plot_pdp_concept(model, X_train, feature_names) -> go.Figure:
    """Overlay 100 ICE curves for 'hr' with the PDP (their average).

    Three vertical guide lines at hr=8, 12, 17 show how the PDP averages over
    the spread of individual curves at each point.
    """
    hr_idx = feature_names.index("hr")
    hr_grid = np.arange(0, 24)

    # Sample 100 training instances for ICE
    samp_idx = RNG.choice(len(X_train), size=100, replace=False)
    X_ice = X_train[samp_idx]
    ice = compute_ice(model, X_ice, hr_idx, hr_grid)
    pdp = ice.mean(axis=0)

    fig = go.Figure()

    # ICE curves (thin, semi-transparent)
    for i in range(len(X_ice)):
        fig.add_trace(go.Scatter(
            x=hr_grid, y=ice[i],
            mode="lines",
            line=dict(color=COLORS["primary"], width=0.8),
            opacity=0.18,
            showlegend=(i == 0),
            name="ICE curve (individual instance)",
            hoverinfo="skip",
        ))

    # PDP = mean of ICE
    fig.add_trace(go.Scatter(
        x=hr_grid, y=pdp,
        mode="lines",
        line=dict(color=COLORS["accent"], width=3.5),
        name="PDP (average of ICE curves)",
        hovertemplate="hr=%{x}<br>Average prediction=%{y:.0f}<extra></extra>",
    ))

    # Vertical guides + spread annotations at hr=8, 12, 17
    guide_hours = [8, 12, 17]
    guide_labels = ["8 AM<br>rush", "Noon", "5 PM<br>rush"]
    for h, lbl in zip(guide_hours, guide_labels):
        col_vals = ice[:, h]
        pdp_val = pdp[h]
        q25, q75 = float(np.percentile(col_vals, 25)), float(np.percentile(col_vals, 75))

        fig.add_shape(
            type="line", x0=h, x1=h, y0=col_vals.min(), y1=col_vals.max(),
            line=dict(color=COLORS["neutral"], width=1.5, dash="dot"),
        )
        # Spread bar (IQR)
        fig.add_trace(go.Scatter(
            x=[h, h], y=[q25, q75],
            mode="lines",
            line=dict(color=COLORS["neutral"], width=4),
            showlegend=False,
            hovertemplate=f"hr={h}<br>IQR: [{q25:.0f}, {q75:.0f}]<extra></extra>",
        ))
        fig.add_annotation(
            x=h, y=pdp_val + 35,
            text=f"{lbl}<br>avg={pdp_val:.0f}",
            showarrow=True, arrowhead=2, ax=0, ay=-30,
            font=dict(size=10, color=COLORS["accent"]),
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor=COLORS["accent"], borderwidth=1,
        )

    fig.update_layout(
        title=(
            "Partial Dependence Plot — Average the Prediction Over All Instances<br>"
            "<sup>PDP (pink) = average of all ICE curves (blue) → the marginal effect of hour on rentals</sup>"
        ),
        xaxis=dict(title="Hour of day (hr)", tickmode="linear", tick0=0, dtick=2),
        yaxis_title="Predicted bike rentals",
        legend=dict(orientation="h", y=-0.18),
        height=520,
        margin=dict(t=90, b=70),
    )
    return fig


# ── Figure 2: 2×2 PDP comparison ──────────────────────────────────────────────

def plot_pdp_comparison(model, X_train, feature_names) -> go.Figure:
    """2×2 subplots: PDP for hr, temp, hum, season with rug plots.

    All y-axes share the same range so steepness is directly comparable across
    features — a feature with a steeper PDP has a stronger marginal effect.
    """
    features = ["hr", "temp", "hum", "season"]
    feat_indices = [feature_names.index(f) for f in features]

    # Grid: 30 points across each feature's range (24 discrete for hr)
    grids = []
    for idx, fname in zip(feat_indices, features):
        if fname == "hr":
            grids.append(np.arange(0, 24))
        elif fname == "season":
            grids.append(np.array([1, 2, 3, 4]))
        else:
            grids.append(np.linspace(X_train[:, idx].min(), X_train[:, idx].max(), 30))

    X_pdp = X_train[:500]   # subsample for speed
    pdps = [compute_pdp(model, X_pdp, idx, grid) for idx, grid in zip(feat_indices, grids)]

    # Shared y range across all panels
    y_min = min(p.min() for p in pdps) * 0.92
    y_max = max(p.max() for p in pdps) * 1.05

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[f"PDP: {f}" for f in features],
        vertical_spacing=0.15,
        horizontal_spacing=0.10,
    )

    positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
    rug_y = y_min + (y_max - y_min) * 0.03   # rug position just above y_min

    for (row, col), fname, feat_idx, grid, pdp in zip(
        positions, features, feat_indices, grids, pdps
    ):
        # PDP line
        mode = "lines" if fname not in ("season",) else "lines+markers"
        fig.add_trace(go.Scatter(
            x=grid, y=pdp,
            mode=mode,
            line=dict(color=COLORS["primary"], width=2.5),
            marker=dict(size=8) if fname == "season" else dict(),
            name=f"PDP({fname})",
            showlegend=False,
            hovertemplate=f"{fname}=%{{x:.2f}}<br>Avg prediction=%{{y:.0f}}<extra></extra>",
        ), row=row, col=col)

        # Rug plot: sample of actual feature values
        rug_x = X_train[RNG.choice(len(X_train), size=400, replace=False), feat_idx]
        fig.add_trace(go.Scatter(
            x=rug_x,
            y=np.full(len(rug_x), rug_y),
            mode="markers",
            marker=dict(
                symbol="line-ns",
                size=6,
                color=COLORS["neutral"],
                opacity=0.25,
                line=dict(color=COLORS["neutral"], width=1),
            ),
            name="Data distribution",
            showlegend=(row == 1 and col == 1),
            hovertemplate=f"{fname}=%{{x:.2f}}<extra></extra>",
        ), row=row, col=col)

        fig.update_yaxes(range=[y_min, y_max], row=row, col=col)
        fig.update_xaxes(title_text=fname, row=row, col=col)

    # Shared y-axis label on the left subplots
    fig.update_yaxes(title_text="Avg predicted rentals", row=1, col=1)
    fig.update_yaxes(title_text="Avg predicted rentals", row=2, col=1)

    fig.update_layout(
        title=(
            "Partial Dependence Plots — Compare Feature Effects Across the Dataset<br>"
            "<sup>Shared y-axis range so slope steepness is directly comparable | "
            "rug marks show data density</sup>"
        ),
        height=520,
        margin=dict(t=90, b=50),
    )
    return fig


# ── Figure 3: 2D PDP heatmap ──────────────────────────────────────────────────

def plot_pdp_2d_concept(model, X_train, feature_names) -> go.Figure:
    """2D PDP heatmap (hr × temp) at the top; 1D PDPs of each feature below.

    If hr and temp had NO interaction, the heatmap would show perfect horizontal
    stripes — every column (fixed temp) would look the same.  Deviations from
    stripes reveal the interaction.
    """
    hr_idx = feature_names.index("hr")
    temp_idx = feature_names.index("temp")

    hr_grid = np.arange(0, 24)
    temp_grid = np.linspace(X_train[:, temp_idx].min(), X_train[:, temp_idx].max(), 20)

    X_pdp = X_train[:300]   # subsample for speed

    print("  Computing 2D PDP (hr × temp) — this may take a moment…")
    z = np.zeros((len(hr_grid), len(temp_grid)))
    for i, hv in enumerate(hr_grid):
        for j, tv in enumerate(temp_grid):
            X_mod = X_pdp.copy()
            X_mod[:, hr_idx] = hv
            X_mod[:, temp_idx] = tv
            z[i, j] = model.predict(X_mod).mean()

    pdp_hr = z.mean(axis=1)       # marginalise over temp
    pdp_temp = z.mean(axis=0)     # marginalise over hr

    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{"colspan": 2}, None], [{}, {}]],
        row_heights=[0.62, 0.38],
        subplot_titles=[
            "2D PDP: hour × temperature",
            "1D PDP: hr (marginalised over temp)",
            "1D PDP: temp (marginalised over hr)",
        ],
        vertical_spacing=0.14,
        horizontal_spacing=0.12,
    )

    # Top: 2D heatmap — y=hr, x=temp
    fig.add_trace(go.Heatmap(
        x=temp_grid,
        y=hr_grid,
        z=z,
        colorscale="RdBu_r",
        colorbar=dict(title="Avg<br>rentals", len=0.55, y=0.72),
        hovertemplate="temp=%{x:.2f}<br>hr=%{y}<br>Avg prediction=%{z:.0f}<extra></extra>",
    ), row=1, col=1)

    fig.update_xaxes(title_text="Temperature (temp)", row=1, col=1)
    fig.update_yaxes(title_text="Hour of day (hr)", row=1, col=1)

    # Bottom-left: 1D PDP of hr
    fig.add_trace(go.Scatter(
        x=hr_grid, y=pdp_hr,
        mode="lines",
        line=dict(color=COLORS["primary"], width=2.5),
        showlegend=False,
        hovertemplate="hr=%{x}<br>Avg=%{y:.0f}<extra></extra>",
    ), row=2, col=1)
    fig.update_xaxes(title_text="hr", tickmode="linear", tick0=0, dtick=4, row=2, col=1)
    fig.update_yaxes(title_text="Avg predicted rentals", row=2, col=1)

    # Bottom-right: 1D PDP of temp
    fig.add_trace(go.Scatter(
        x=temp_grid, y=pdp_temp,
        mode="lines",
        line=dict(color=COLORS["accent"], width=2.5),
        showlegend=False,
        hovertemplate="temp=%{x:.2f}<br>Avg=%{y:.0f}<extra></extra>",
    ), row=2, col=2)
    fig.update_xaxes(title_text="temp", row=2, col=2)
    fig.update_yaxes(title_text="Avg predicted rentals", row=2, col=2)

    # Annotation on heatmap: explain interaction vs no-interaction
    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.5, y=0.62,
        text=(
            "If hr and temp had <b>NO interaction</b>, this heatmap would show<br>"
            "uniform horizontal stripes — same pattern regardless of temperature."
        ),
        showarrow=False, align="center",
        font=dict(size=11, color="#111827"),
        bgcolor="rgba(255,255,255,0.88)",
        bordercolor=COLORS["neutral"], borderwidth=1,
    )

    fig.update_layout(
        title=(
            "2D Partial Dependence — Do Two Features Interact?<br>"
            "<sup>Non-horizontal stripes in the heatmap signal that hr and temp interact</sup>"
        ),
        height=560,
        margin=dict(t=90, b=40, r=80),
    )
    return fig


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    X_train, X_test, y_train, y_test, feature_names = load_regression()
    model = get_regression_model(X_train, y_train)

    print("Figure 1: ICE curves + PDP concept…")
    fig1 = plot_pdp_concept(model, X_train, feature_names)
    save_figure(fig1, CHAPTER, "how_pdp_concept")

    print("Figure 2: 2×2 PDP comparison…")
    fig2 = plot_pdp_comparison(model, X_train, feature_names)
    save_figure(fig2, CHAPTER, "how_pdp_comparison")

    print("Figure 3: 2D PDP heatmap (hr × temp)…")
    fig3 = plot_pdp_2d_concept(model, X_train, feature_names)
    save_figure(fig3, CHAPTER, "how_pdp_2d_concept")


if __name__ == "__main__":
    main()

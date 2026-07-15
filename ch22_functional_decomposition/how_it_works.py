"""
Chapter 22: How Functional Decomposition Works
https://christophm.github.io/interpretable-ml-book/decomposition.html

Conceptual figures explaining the method — separate from model-output figures in main.py.

Figures produced:
  1. how_decomp_concept.html/png          — main effects of hr and temp + residual scatter
  2. how_decomp_variance.html/png         — variance explained by each component
  3. how_decomp_interaction_check.html/png — residual vs features to reveal interactions
"""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from shared.datasets import load_regression
from shared.models import get_regression_model
from shared.theme import COLORS, save_figure

CHAPTER = "ch22_functional_decomposition"
RNG = np.random.default_rng(42)
N_GRID = 50


# ── Helpers ───────────────────────────────────────────────────────────────────

def compute_main_effect(model, X, feature_idx, n_grid=N_GRID):
    """Centered PDP: vary one feature, hold all others at their column mean,
    subtract the overall mean prediction so the effect is centered at zero."""
    grid = np.linspace(X[:, feature_idx].min(), X[:, feature_idx].max(), n_grid)
    X_mean = X.mean(axis=0)
    preds = []
    for v in grid:
        Xc = np.tile(X_mean, (len(X), 1))
        Xc[:, feature_idx] = v
        preds.append(model.predict(Xc).mean())
    arr = np.array(preds)
    return grid, arr - arr.mean()


def assign_main_effect(grid, effect, feature_vals):
    """Map each instance's feature value to its nearest grid-point main effect."""
    indices = np.searchsorted(grid, feature_vals).clip(0, len(grid) - 1)
    return effect[indices]


# ── Figure 1: Decomposition concept ──────────────────────────────────────────

def plot_decomp_concept(model, X_train, X_test, y_test, feature_names) -> go.Figure:
    hr_idx = feature_names.index("hr")
    temp_idx = feature_names.index("temp")

    grid_hr, effect_hr = compute_main_effect(model, X_train, hr_idx)
    grid_temp, effect_temp = compute_main_effect(model, X_train, temp_idx)

    mean_pred = model.predict(X_train).mean()

    # 200 test instances for the residual panel
    idx200 = RNG.choice(len(X_test), size=200, replace=False)
    X200 = X_test[idx200]
    preds200 = model.predict(X200)
    me_hr200 = assign_main_effect(grid_hr, effect_hr, X200[:, hr_idx])
    me_temp200 = assign_main_effect(grid_temp, effect_temp, X200[:, temp_idx])
    residual200 = preds200 - mean_pred - me_hr200 - me_temp200

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=["Main effect: hr", "Main effect: temp", "Residual (interactions + noise)"],
        horizontal_spacing=0.10,
    )

    # Left — main effect of hr
    fig.add_trace(go.Scatter(
        x=grid_hr, y=effect_hr, mode="lines",
        line=dict(color=COLORS["primary"], width=2.5),
        name="Main effect: hr",
        hovertemplate="hr=%{x:.1f}<br>Effect=%{y:.1f}<extra></extra>",
    ), row=1, col=1)
    fig.add_hline(y=0, line_dash="dot", line_color=COLORS["neutral"], row=1, col=1)

    # Middle — main effect of temp
    fig.add_trace(go.Scatter(
        x=grid_temp, y=effect_temp, mode="lines",
        line=dict(color=COLORS["accent"], width=2.5),
        name="Main effect: temp",
        hovertemplate="temp=%{x:.2f}<br>Effect=%{y:.1f}<extra></extra>",
    ), row=1, col=2)
    fig.add_hline(y=0, line_dash="dot", line_color=COLORS["neutral"], row=1, col=2)

    # Right — residual scatter vs model prediction
    fig.add_trace(go.Scatter(
        x=preds200, y=residual200, mode="markers",
        marker=dict(color=COLORS["neutral"], size=5, opacity=0.55),
        name="Residual",
        hovertemplate="Prediction=%{x:.0f}<br>Residual=%{y:.1f}<extra></extra>",
    ), row=1, col=3)
    fig.add_hline(y=0, line_dash="dot", line_color=COLORS["neutral"], row=1, col=3)

    fig.update_xaxes(title_text="Hour of day", row=1, col=1)
    fig.update_yaxes(title_text="Effect on prediction (centered)", row=1, col=1)
    fig.update_xaxes(title_text="Temperature (normalised)", row=1, col=2)
    fig.update_xaxes(title_text="Model prediction", row=1, col=3)
    fig.update_yaxes(title_text="Residual", row=1, col=3)

    fig.update_layout(
        title=(
            "Functional Decomposition — A Model Is a Sum of Components<br>"
            "<sup>f(x) = μ + f<sub>hr</sub>(hr) + f<sub>temp</sub>(temp) + "
            "interactions + noise</sup>"
        ),
        height=500,
        showlegend=False,
    )
    return fig


# ── Figure 2: Variance explained by each component ───────────────────────────

def plot_decomp_variance(model, X_train, feature_names) -> go.Figure:
    importances = model.feature_importances_
    top8_idx = list(np.argsort(importances)[-8:][::-1])

    f_hat = model.predict(X_train)
    total_var = float(np.var(f_hat))

    fractions = []
    for fi in top8_idx:
        grid, effect = compute_main_effect(model, X_train, fi)
        me_instances = assign_main_effect(grid, effect, X_train[:, fi])
        fractions.append(float(np.var(me_instances)) / (total_var + 1e-10))

    residual_frac = max(0.0, 1.0 - sum(fractions))

    names = [feature_names[i] for i in top8_idx] + ["Residual (interactions)"]
    fracs = fractions + [residual_frac]
    order = np.argsort(fracs)

    sorted_names = [names[i] for i in order]
    sorted_fracs = [fracs[i] for i in order]
    sorted_colors = [
        COLORS["neutral"] if n == "Residual (interactions)" else COLORS["primary"]
        for n in sorted_names
    ]

    # Find the top feature for the annotation
    top_name = feature_names[top8_idx[0]]
    top_frac = fractions[0]

    fig = go.Figure(go.Bar(
        x=sorted_fracs,
        y=sorted_names,
        orientation="h",
        marker_color=sorted_colors,
        text=[f"{v:.1%}" for v in sorted_fracs],
        textposition="outside",
        hovertemplate="%{y}: %{x:.2%}<extra></extra>",
    ))

    fig.add_annotation(
        x=top_frac * 0.5,
        y=len(sorted_names) - 1,
        text=f"'{top_name}' alone explains<br>{top_frac:.1%} of model variance",
        showarrow=True,
        arrowhead=2,
        ax=60, ay=-30,
        font=dict(size=12, color=COLORS["primary"]),
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor=COLORS["primary"],
        borderwidth=1,
    )

    fig.update_layout(
        title=(
            "Functional Decomposition — How Much Variance Does Each Component Explain?<br>"
            "<sup>Fraction of model prediction variance attributed to each feature's main effect</sup>"
        ),
        xaxis=dict(title="Fraction of prediction variance", tickformat=".0%"),
        height=520,
        legend=dict(orientation="h", y=-0.18),
    )
    return fig


# ── Figure 3: Interaction check ───────────────────────────────────────────────

def plot_decomp_interaction_check(model, X_train, X_test, feature_names) -> go.Figure:
    importances = model.feature_importances_
    top8_idx = list(np.argsort(importances)[-8:][::-1])

    hr_idx = feature_names.index("hr")
    temp_idx = feature_names.index("temp")
    season_idx = feature_names.index("season")

    mean_pred = model.predict(X_train).mean()

    idx500 = RNG.choice(len(X_test), size=min(500, len(X_test)), replace=False)
    X500 = X_test[idx500]
    preds500 = model.predict(X500)

    # Residual = prediction - mean - sum of all top-8 main effects
    residual = preds500 - mean_pred
    for fi in top8_idx:
        grid, effect = compute_main_effect(model, X_train, fi)
        residual = residual - assign_main_effect(grid, effect, X500[:, fi])

    seasons = X500[:, season_idx].astype(int)
    season_names = {1: "Spring", 2: "Summer", 3: "Fall", 4: "Winter"}

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Residual vs hr", "Residual vs temp"],
        horizontal_spacing=0.12,
    )

    for season_val, sname in season_names.items():
        mask = seasons == season_val
        if not mask.any():
            continue
        color = COLORS["palette"][season_val - 1]

        fig.add_trace(go.Scatter(
            x=X500[mask, hr_idx], y=residual[mask], mode="markers",
            marker=dict(color=color, size=5, opacity=0.55),
            name=sname,
            legendgroup=sname,
            hovertemplate=f"{sname}<br>hr=%{{x:.0f}}<br>Residual=%{{y:.1f}}<extra></extra>",
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=X500[mask, temp_idx], y=residual[mask], mode="markers",
            marker=dict(color=color, size=5, opacity=0.55),
            name=sname,
            legendgroup=sname,
            showlegend=False,
            hovertemplate=f"{sname}<br>temp=%{{x:.2f}}<br>Residual=%{{y:.1f}}<extra></extra>",
        ), row=1, col=2)

    fig.add_hline(y=0, line_dash="dot", line_color=COLORS["neutral"], row=1, col=1)
    fig.add_hline(y=0, line_dash="dot", line_color=COLORS["neutral"], row=1, col=2)

    fig.update_xaxes(title_text="Hour of day", row=1, col=1)
    fig.update_yaxes(title_text="Residual (after removing main effects)", row=1, col=1)
    fig.update_xaxes(title_text="Temperature (normalised)", row=1, col=2)
    fig.update_yaxes(title_text="Residual", row=1, col=2)

    fig.update_layout(
        title=(
            "Functional Decomposition — Checking for Interactions in the Residual<br>"
            "<sup>A pattern here means the main effects didn't capture everything — "
            "interactions remain</sup>"
        ),
        height=510,
        legend=dict(orientation="h", y=-0.18),
    )
    return fig


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    X_train, X_test, y_train, y_test, feature_names = load_regression()
    model = get_regression_model(X_train, y_train)

    print("Figure 1: Decomposition concept…")
    fig1 = plot_decomp_concept(model, X_train, X_test, y_test, feature_names)
    save_figure(fig1, CHAPTER, "how_decomp_concept")

    print("Figure 2: Variance explained by each component…")
    fig2 = plot_decomp_variance(model, X_train, feature_names)
    save_figure(fig2, CHAPTER, "how_decomp_variance")

    print("Figure 3: Interaction check…")
    fig3 = plot_decomp_interaction_check(model, X_train, X_test, feature_names)
    save_figure(fig3, CHAPTER, "how_decomp_interaction_check")


if __name__ == "__main__":
    main()

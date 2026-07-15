"""
Chapter 23: How Permutation Importance Works
https://christophm.github.io/interpretable-ml-book/feature-importance.html

Conceptual figures explaining the method — separate from model-output figures in main.py.

Figures produced:
  1. how_perm_concept.html/png      — structured vs permuted scatter for hr
  2. how_perm_distribution.html/png — violin plots of R² drop over 20 repeats
  3. how_perm_vs_impurity.html/png  — MDI vs permutation importance comparison
"""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.inspection import permutation_importance

from shared.datasets import load_regression
from shared.models import get_regression_model
from shared.theme import COLORS, save_figure

CHAPTER = "ch23_permutation_importance"
RNG = np.random.default_rng(42)


# ── Figure 1: Before vs after permutation ────────────────────────────────────

def plot_perm_concept(model, X_test, y_test, feature_names) -> go.Figure:
    hr_idx = feature_names.index("hr")

    idx500 = RNG.choice(len(X_test), size=500, replace=False)
    X500 = X_test[idx500]
    y500 = y_test[idx500]
    preds500 = model.predict(X500)

    # Shuffled version of hr
    hr_shuffled = X500[:, hr_idx].copy()
    RNG.shuffle(hr_shuffled)

    # Shared color scale (actual prediction) — no colorbar
    cmin, cmax = float(preds500.min()), float(preds500.max())
    marker_before = dict(
        color=preds500, colorscale="Viridis",
        cmin=cmin, cmax=cmax,
        size=5, opacity=0.6, showscale=False,
    )
    marker_after = dict(
        color=preds500, colorscale="Viridis",
        cmin=cmin, cmax=cmax,
        size=5, opacity=0.6, showscale=False,
    )

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[
            "Before: structured relationship",
            "After permutation: random",
        ],
        horizontal_spacing=0.12,
    )

    fig.add_trace(go.Scatter(
        x=X500[:, hr_idx], y=y500, mode="markers",
        marker=marker_before,
        name="Before",
        hovertemplate="hr=%{x:.0f}<br>Rentals=%{y:.0f}<extra></extra>",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=hr_shuffled, y=y500, mode="markers",
        marker=marker_after,
        name="After permutation",
        hovertemplate="hr (shuffled)=%{x:.0f}<br>Rentals=%{y:.0f}<extra></extra>",
    ), row=1, col=2)

    fig.update_xaxes(title_text="Hour of day", row=1, col=1)
    fig.update_yaxes(title_text="Bike rentals", row=1, col=1)
    fig.update_xaxes(title_text="Hour of day (shuffled)", row=1, col=2)
    fig.update_yaxes(title_text="Bike rentals", row=1, col=2)

    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.5, y=-0.12,
        text=(
            "If permuting a feature degrades model performance → it was important.<br>"
            "The left panel shows structure; the right shows pure noise after shuffling."
        ),
        showarrow=False,
        font=dict(size=12, color=COLORS["neutral"]),
        align="center",
    )

    fig.update_layout(
        title=(
            "Permutation Importance — Break a Feature's Relationship with the Target<br>"
            "<sup>Shuffling 'hr' destroys its predictive signal; "
            "if performance drops, hr was important</sup>"
        ),
        height=500,
        showlegend=False,
        margin=dict(b=100),
    )
    return fig


# ── Figure 2: Distribution of performance drops ───────────────────────────────

def plot_perm_distribution(model, X_test, y_test, feature_names) -> go.Figure:
    # Top 5 features by MDI
    top5_idx = list(np.argsort(model.feature_importances_)[-5:])
    top5_names = [feature_names[i] for i in top5_idx]

    # Permutation importance with 20 repeats for all top-5 features
    result = permutation_importance(
        model, X_test, y_test,
        n_repeats=20,
        random_state=42,
        scoring="r2",
    )

    # Extract importances for top-5 and sort by median descending
    top5_importances = result.importances[top5_idx]  # shape (5, 20)
    medians = np.median(top5_importances, axis=1)
    order = np.argsort(medians)[::-1]  # descending

    sorted_names = [top5_names[i] for i in order]
    sorted_imp = [top5_importances[i] for i in order]

    fig = go.Figure()

    for name, imp_vals in zip(sorted_names, sorted_imp):
        fig.add_trace(go.Violin(
            y=imp_vals,
            x=[name] * len(imp_vals),
            name=name,
            box_visible=True,
            meanline_visible=True,
            fillcolor=COLORS["primary"],
            opacity=0.7,
            line_color=COLORS["primary"],
            showlegend=False,
            hovertemplate=f"{name}<br>ΔR²=%{{y:.4f}}<extra></extra>",
        ))

    # Horizontal reference at 0
    fig.add_hline(
        y=0, line_dash="dash", line_color=COLORS["neutral"], line_width=1.5,
        annotation_text="No effect (ΔR² = 0)",
        annotation_position="right",
    )

    # Annotate most important feature
    top_name = sorted_names[0]
    top_vals = sorted_imp[0]
    top_mean = float(np.mean(top_vals))
    top_std = float(np.std(top_vals))
    fig.add_annotation(
        x=top_name,
        y=float(np.max(top_vals)) * 1.05,
        text=f"Shuffling '{top_name}'<br>drops R² by {top_mean:.3f} ± {top_std:.3f}",
        showarrow=True,
        arrowhead=2,
        ax=60, ay=-30,
        font=dict(size=12, color=COLORS["accent"]),
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor=COLORS["accent"],
        borderwidth=1,
    )

    fig.update_layout(
        title=(
            "Permutation Importance — Repeat Shuffles to Estimate Uncertainty<br>"
            "<sup>20 independent permutations per feature; wider violin = more variable importance</sup>"
        ),
        xaxis_title="Feature",
        yaxis_title="Decrease in R² when feature is permuted",
        height=510,
        legend=dict(orientation="h", y=-0.18),
    )
    return fig


# ── Figure 3: Permutation vs MDI impurity ────────────────────────────────────

def plot_perm_vs_impurity(model, X_test, y_test, feature_names) -> go.Figure:
    mdi = model.feature_importances_

    result = permutation_importance(
        model, X_test, y_test,
        n_repeats=10,
        random_state=42,
        scoring="r2",
    )
    perm = result.importances_mean

    # Sort each panel independently by its own values
    order_mdi = np.argsort(mdi)
    order_perm = np.argsort(perm)

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Impurity (MDI)", "Permutation (actual R² drop)"],
        horizontal_spacing=0.18,
    )

    fig.add_trace(go.Bar(
        x=mdi[order_mdi],
        y=[feature_names[i] for i in order_mdi],
        orientation="h",
        marker_color=COLORS["primary"],
        name="MDI",
        hovertemplate="%{y}: %{x:.4f}<extra></extra>",
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=perm[order_perm],
        y=[feature_names[i] for i in order_perm],
        orientation="h",
        marker_color=COLORS["accent"],
        name="Permutation",
        hovertemplate="%{y}: %{x:.4f}<extra></extra>",
    ), row=1, col=2)

    fig.update_xaxes(title_text="MDI importance (unitless)", row=1, col=1)
    fig.update_xaxes(title_text="Mean decrease in R²", row=1, col=2)
    fig.add_vline(x=0, line_color=COLORS["neutral"], line_width=1, row=1, col=2)

    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.5, y=-0.12,
        text="MDI can overstate importance of high-cardinality features; "
             "permutation reflects actual predictive contribution.",
        showarrow=False,
        font=dict(size=12, color=COLORS["neutral"]),
        align="center",
    )

    fig.update_layout(
        title=(
            "Permutation Importance vs Impurity Importance — Two Different Stories<br>"
            "<sup>MDI is fast but biased; permutation importance is model-agnostic</sup>"
        ),
        height=510,
        legend=dict(orientation="h", y=-0.18),
        margin=dict(b=90),
    )
    return fig


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    X_train, X_test, y_train, y_test, feature_names = load_regression()
    model = get_regression_model(X_train, y_train)

    print("Figure 1: Before/after permutation concept…")
    fig1 = plot_perm_concept(model, X_test, y_test, feature_names)
    save_figure(fig1, CHAPTER, "how_perm_concept")

    print("Figure 2: Distribution of R² drops over 20 repeats…")
    fig2 = plot_perm_distribution(model, X_test, y_test, feature_names)
    save_figure(fig2, CHAPTER, "how_perm_distribution")

    print("Figure 3: Permutation vs MDI impurity comparison…")
    fig3 = plot_perm_vs_impurity(model, X_test, y_test, feature_names)
    save_figure(fig3, CHAPTER, "how_perm_vs_impurity")


if __name__ == "__main__":
    main()

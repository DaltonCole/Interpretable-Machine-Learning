"""
Chapter 12: How Ceteris Paribus Profiles Work
https://christophm.github.io/interpretable-ml-book/ceteris-paribus.html

Conceptual figures explaining the method — separate from the model-output
figures in main.py.

Figures produced:
  1. how_cp_concept.html/png       — table + CP line; shows the core "freeze-one" idea
  2. how_cp_multifeature.html/png  — 2×3 grid of CP profiles for top-6 features
  3. how_cp_comparison.html/png    — 5 diverse instances overlaid on the hr CP profile
"""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from shared.datasets import load_regression
from shared.models import get_regression_model
from shared.theme import COLORS, save_figure

CHAPTER = "ch12_ceteris_paribus"
RNG = np.random.default_rng(42)


# ── Helpers ───────────────────────────────────────────────────────────────────

def compute_cp_profile(model, instance, X_train, feature_idx, n_points=100):
    """Vary feature_idx across its training range; freeze everything else."""
    grid = np.linspace(X_train[:, feature_idx].min(),
                       X_train[:, feature_idx].max(), n_points)
    X_sweep = np.tile(instance, (n_points, 1))
    X_sweep[:, feature_idx] = grid
    return grid, model.predict(X_sweep)


def pick_instance(model, X_test, feature_names):
    """Return a test-set instance with hr≈8 and a high prediction (morning rush)."""
    hr_idx = feature_names.index("hr")
    preds = model.predict(X_test)
    hr_vals = X_test[:, hr_idx]
    mask = (hr_vals >= 7) & (hr_vals <= 9)
    if mask.any():
        return int(np.where(mask)[0][int(np.argmax(preds[mask]))])
    return int(np.argmax(preds))


# ── Figure 1: core concept ────────────────────────────────────────────────────

def plot_cp_concept(model, instance, X_train, feature_names) -> go.Figure:
    """Left: feature table with hr highlighted.
    Right: CP profile for hr with actual-value marker."""
    hr_idx = feature_names.index("hr")
    grid, cp_preds = compute_cp_profile(model, instance, X_train, hr_idx, n_points=24)
    actual_hr = float(instance[hr_idx])
    original_pred = float(model.predict(instance.reshape(1, -1))[0])

    # ── layout ──
    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.42, 0.58],
        specs=[[{"type": "table"}, {"type": "scatter"}]],
        subplot_titles=["Instance Feature Values", "CP Profile: hour of day (hr)"],
        horizontal_spacing=0.06,
    )

    # ── left: feature table ──
    feat_vals = [f"{v:.1f}" for v in instance]
    notes = ["← this one changes" if fn == "hr" else "" for fn in feature_names]
    cell_fill = [
        "rgba(37,99,235,0.14)" if fn == "hr" else "white"
        for fn in feature_names
    ]
    name_col = [f"<b>{fn}</b>" if fn == "hr" else fn for fn in feature_names]

    fig.add_trace(go.Table(
        header=dict(
            values=["<b>Feature</b>", "<b>Value</b>", "<b>Note</b>"],
            fill_color=COLORS["primary"],
            font=dict(color="white", size=12),
            align="left",
            height=28,
        ),
        cells=dict(
            values=[name_col, feat_vals, notes],
            fill_color=[list(cell_fill), list(cell_fill), list(cell_fill)],
            align=["left", "right", "left"],
            font=dict(size=11, color="#1F2937"),
            height=24,
        ),
    ), row=1, col=1)

    # ── right: CP profile line ──
    fig.add_trace(go.Scatter(
        x=grid, y=cp_preds,
        mode="lines",
        line=dict(color=COLORS["primary"], width=3),
        name="CP profile (varying hr)",
        hovertemplate="hr=%{x:.0f}<br>Predicted=%{y:.0f}<extra></extra>",
    ), row=1, col=2)

    # vertical dashed line at actual hr
    fig.add_trace(go.Scatter(
        x=[actual_hr, actual_hr],
        y=[float(cp_preds.min()) - 30, float(cp_preds.max()) + 30],
        mode="lines",
        line=dict(color=COLORS["neutral"], dash="dash", width=1.5),
        name=f"Actual hr = {actual_hr:.0f}",
        showlegend=True,
    ), row=1, col=2)

    # dot at actual prediction
    fig.add_trace(go.Scatter(
        x=[actual_hr],
        y=[original_pred],
        mode="markers",
        marker=dict(size=13, color=COLORS["accent"], symbol="circle",
                    line=dict(color="white", width=2)),
        name=f"Actual prediction ({original_pred:.0f})",
    ), row=1, col=2)

    # annotation
    fig.add_annotation(
        x=0.98, y=0.97,
        xref="paper", yref="paper",
        text="Everything else stays fixed.<br>Only <b>hr</b> moves.",
        showarrow=False,
        align="right", xanchor="right", yanchor="top",
        font=dict(size=12),
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor=COLORS["primary"], borderwidth=1,
    )

    fig.update_xaxes(title_text="Hour of day", tickmode="linear",
                     tick0=0, dtick=4, row=1, col=2)
    fig.update_yaxes(title_text="Predicted bike rentals", row=1, col=2)

    fig.update_layout(
        title=("Ceteris Paribus — Vary One Feature, Freeze Everything Else<br>"
               "<sup>Model prediction as 'hr' sweeps 0→23; all other feature values frozen at one instance.</sup>"),
        height=500,
        legend=dict(orientation="h", y=-0.18),
    )
    return fig


# ── Figure 2: multi-feature CP profiles ──────────────────────────────────────

def plot_cp_multifeature(model, instance, X_train, feature_names) -> go.Figure:
    """2×3 grid of CP profiles for the top-6 features by importance.

    All subplots share the same y-axis range so steepness is directly comparable.
    """
    importances = model.feature_importances_
    top6_idx = np.argsort(importances)[-6:][::-1]  # most important first
    original_pred = float(model.predict(instance.reshape(1, -1))[0])

    # Compute all profiles and find global y range
    profiles = []
    for feat_idx in top6_idx:
        grid, preds = compute_cp_profile(model, instance, X_train, feat_idx, n_points=80)
        profiles.append((grid, preds, feat_idx))

    all_preds = np.concatenate([p for _, p, _ in profiles])
    margin = (all_preds.max() - all_preds.min()) * 0.06
    yaxis_range = [float(all_preds.min()) - margin, float(all_preds.max()) + margin]

    subplot_titles = [feature_names[i] for i in top6_idx]
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=subplot_titles,
        vertical_spacing=0.14,
        horizontal_spacing=0.09,
    )

    for pos, (grid, preds, feat_idx) in enumerate(profiles):
        r, c = divmod(pos, 3)
        r += 1; c += 1

        # CP line
        fig.add_trace(go.Scatter(
            x=grid, y=preds,
            mode="lines",
            line=dict(color=COLORS["primary"], width=2),
            showlegend=False,
            hovertemplate=f"{feature_names[feat_idx]}=%{{x:.2f}}<br>Predicted=%{{y:.0f}}<extra></extra>",
        ), row=r, col=c)

        # horizontal line at original prediction
        fig.add_shape(
            type="line",
            x0=float(grid.min()), x1=float(grid.max()),
            y0=original_pred, y1=original_pred,
            line=dict(color=COLORS["neutral"], dash="dash", width=1),
            row=r, col=c,
        )

        # red dot at actual feature value
        actual_val = float(instance[feat_idx])
        fig.add_trace(go.Scatter(
            x=[actual_val],
            y=[original_pred],
            mode="markers",
            marker=dict(size=10, color=COLORS["negative"],
                        line=dict(color="white", width=1.5)),
            showlegend=pos == 0,
            name="Actual value",
        ), row=r, col=c)

    # Add legend entry for the dashed line via a dummy trace
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="lines",
        line=dict(color=COLORS["neutral"], dash="dash", width=1),
        name="Original prediction",
    ))

    # Synchronize y-axis across all subplots
    fig.update_yaxes(range=yaxis_range)

    fig.update_layout(
        title=("Ceteris Paribus — How Sensitive Is This Instance to Each Feature?<br>"
               "<sup>Steeper curve = instance is more sensitive to that feature. "
               "Y-axis synchronized — shapes are directly comparable.</sup>"),
        height=540,
        legend=dict(orientation="h", y=-0.12),
    )
    return fig


# ── Figure 3: cross-instance comparison ──────────────────────────────────────

def plot_cp_comparison(model, X_test, X_train, feature_names) -> go.Figure:
    """5 diverse instances' CP profiles for hr overlaid on one plot.

    Shows heterogeneity: some instances respond more to hour than others.
    """
    hr_idx = feature_names.index("hr")
    preds_all = model.predict(X_test)
    sorted_idx = np.argsort(preds_all)
    n = len(sorted_idx)

    # Sample 5 instances spanning the prediction distribution
    quantiles = [0.05, 0.25, 0.50, 0.75, 0.95]
    chosen_idx = [sorted_idx[int(n * q)] for q in quantiles]

    fig = go.Figure()

    for k, inst_idx in enumerate(chosen_idx):
        inst = X_test[inst_idx]
        pred = float(preds_all[inst_idx])
        grid, cp_preds = compute_cp_profile(model, inst, X_train, hr_idx, n_points=24)

        color = COLORS["palette"][k % len(COLORS["palette"])]
        actual_hr = float(inst[hr_idx])

        # CP line
        fig.add_trace(go.Scatter(
            x=grid, y=cp_preds,
            mode="lines",
            line=dict(color=color, width=2.2),
            name=f"Instance {k+1} (pred={pred:.0f})",
            hovertemplate=f"Instance {k+1} — hr=%{{x:.0f}}, pred=%{{y:.0f}}<extra></extra>",
        ))

        # dot at actual hr value
        fig.add_trace(go.Scatter(
            x=[actual_hr],
            y=[pred],
            mode="markers",
            marker=dict(size=11, color=color, symbol="circle",
                        line=dict(color="white", width=2)),
            showlegend=False,
            hovertemplate=f"Instance {k+1}: hr={actual_hr:.0f}, pred={pred:.0f}<extra></extra>",
        ))

    fig.add_annotation(
        x=0.5, y=0.97,
        xref="paper", yref="paper",
        text="Dots mark each instance's actual hr value and current prediction.",
        showarrow=False, align="center", xanchor="center", yanchor="top",
        font=dict(size=12),
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor=COLORS["neutral"], borderwidth=1,
    )

    fig.update_layout(
        title=("Ceteris Paribus — Same Feature, Different Instances React Differently<br>"
               "<sup>Each line = one instance's CP profile for 'hr'. "
               "Steep line = strong response to hour; flat = insensitive.</sup>"),
        xaxis=dict(title="Hour of day", tickmode="linear", tick0=0, dtick=4),
        yaxis_title="Predicted bike rentals",
        height=500,
        legend=dict(orientation="h", y=-0.18),
    )
    return fig


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    X_train, X_test, y_train, y_test, feature_names = load_regression()
    model = get_regression_model(X_train, y_train)

    inst_idx = pick_instance(model, X_test, feature_names)
    instance = X_test[inst_idx]
    pred = float(model.predict(instance.reshape(1, -1))[0])
    print(f"Chosen instance idx={inst_idx}, hr={instance[feature_names.index('hr')]:.0f}, "
          f"pred={pred:.0f}")

    print("Figure 1: CP concept (table + profile)…")
    fig1 = plot_cp_concept(model, instance, X_train, feature_names)
    save_figure(fig1, CHAPTER, "how_cp_concept")

    print("Figure 2: Multi-feature CP grid…")
    fig2 = plot_cp_multifeature(model, instance, X_train, feature_names)
    save_figure(fig2, CHAPTER, "how_cp_multifeature")

    print("Figure 3: Cross-instance CP comparison…")
    fig3 = plot_cp_comparison(model, X_test, X_train, feature_names)
    save_figure(fig3, CHAPTER, "how_cp_comparison")


if __name__ == "__main__":
    main()

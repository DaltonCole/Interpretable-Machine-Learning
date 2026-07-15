"""
Chapter 15: How Counterfactual Explanations Work
https://christophm.github.io/interpretable-ml-book/counterfactual.html

No dice_ml dependency — counterfactuals are found by searching the training
set for the closest high-prediction instance.

Figures produced:
  1. how_cf_concept.html/png  — 2D scatter (hr, temp) with original + CF + arrow
  2. how_cf_changes.html/png  — delta bar chart: what exactly needs to change?
  3. how_cf_diverse.html/png  — parallel coordinates for 4 diverse counterfactuals
"""
import numpy as np
import plotly.graph_objects as go

from shared.datasets import load_regression
from shared.models import get_regression_model
from shared.theme import COLORS, save_figure

CHAPTER = "ch15_counterfactual"
RNG = np.random.default_rng(42)

TARGET_LIFT = 150   # we want CF prediction ≥ original + TARGET_LIFT


# ── Helpers ───────────────────────────────────────────────────────────────────

def pick_instance(model, X_test):
    """Return index of a test instance with a moderate prediction (≈150–300)."""
    preds = model.predict(X_test)
    mask = (preds >= 100) & (preds <= 300)
    if mask.any():
        return int(np.where(mask)[0][int(np.argmin(np.abs(preds[mask] - 200)))])
    return int(np.argmin(np.abs(preds - 200)))


def find_cf(model, X_train, instance, exclude_cols=None):
    """Return the training instance closest to `instance` (standardised L2)
    among those with prediction ≥ instance_pred + TARGET_LIFT.

    exclude_cols: list of feature indices to exclude from distance computation
    (i.e., those features are allowed to change freely).
    """
    train_preds = model.predict(X_train)
    inst_pred = float(model.predict(instance.reshape(1, -1))[0])
    threshold = inst_pred + TARGET_LIFT

    mask = train_preds >= threshold
    if not mask.any():
        threshold = inst_pred + TARGET_LIFT * 0.7
        mask = train_preds >= threshold
    if not mask.any():
        mask = np.ones(len(X_train), dtype=bool)

    X_cand = X_train[mask]
    std = X_train.std(axis=0) + 1e-8

    all_cols = list(range(X_train.shape[1]))
    dist_cols = [c for c in all_cols if c not in (exclude_cols or [])]

    dists = np.linalg.norm(
        (X_cand[:, dist_cols] - instance[dist_cols]) / std[dist_cols],
        axis=1,
    )
    best = int(np.argmin(dists))
    return X_cand[best], float(model.predict(X_cand[[best]])[0])


# ── Figure 1: 2D concept scatter ─────────────────────────────────────────────

def plot_cf_concept(model, instance, original_pred, cf, cf_pred,
                    X_train, feature_names) -> go.Figure:
    """Scatter in (hr, temp) 2D coloured by rental count.
    Original and counterfactual marked with arrow between them."""
    hr_idx = feature_names.index("hr")
    temp_idx = feature_names.index("temp")

    samp_idx = RNG.choice(len(X_train), size=500, replace=False)
    X_samp = X_train[samp_idx]
    y_samp = model.predict(X_samp)

    fig = go.Figure()

    # Background scatter
    fig.add_trace(go.Scatter(
        x=X_samp[:, hr_idx], y=X_samp[:, temp_idx],
        mode="markers",
        marker=dict(
            color=y_samp,
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title="Rentals", thickness=14, len=0.75),
            size=5,
            opacity=0.45,
        ),
        name="Training instances",
        hovertemplate="hr=%{x:.0f}<br>temp=%{y:.2f}<br>pred=%{marker.color:.0f}<extra></extra>",
    ))

    # Arrow from original to counterfactual
    fig.add_annotation(
        x=float(cf[hr_idx]),
        y=float(cf[temp_idx]),
        ax=float(instance[hr_idx]),
        ay=float(instance[temp_idx]),
        axref="x", ayref="y",
        text="",
        showarrow=True,
        arrowhead=3,
        arrowcolor=COLORS["accent"],
        arrowwidth=2.5,
    )

    # Original instance
    fig.add_trace(go.Scatter(
        x=[instance[hr_idx]], y=[instance[temp_idx]],
        mode="markers+text",
        marker=dict(size=18, color=COLORS["primary"], symbol="circle",
                    line=dict(color="white", width=2.5)),
        text=[f"Original<br>({original_pred:.0f})"],
        textposition="bottom center",
        textfont=dict(size=12, color=COLORS["primary"]),
        name=f"Original (pred={original_pred:.0f})",
    ))

    # Counterfactual
    fig.add_trace(go.Scatter(
        x=[cf[hr_idx]], y=[cf[temp_idx]],
        mode="markers+text",
        marker=dict(size=18, color=COLORS["accent"], symbol="star",
                    line=dict(color="white", width=2)),
        text=[f"Counterfactual<br>({cf_pred:.0f})"],
        textposition="top center",
        textfont=dict(size=12, color=COLORS["accent"]),
        name=f"Counterfactual (pred={cf_pred:.0f})",
    ))

    fig.update_layout(
        title=("Counterfactual — The Minimal Change That Would Change the Outcome<br>"
               "<sup>Same spot in feature space, very different prediction. "
               f"Arrow shows the 'jump' from {original_pred:.0f} → {cf_pred:.0f} rentals.</sup>"),
        xaxis=dict(title="Hour of day (hr)", tickmode="linear", tick0=0, dtick=4),
        yaxis_title="Normalised temperature (temp)",
        height=520,
        legend=dict(orientation="h", y=-0.18),
    )
    return fig


# ── Figure 2: delta bar chart ─────────────────────────────────────────────────

def plot_cf_changes(instance, cf, original_pred, cf_pred, feature_names) -> go.Figure:
    """Horizontal bar chart of counterfactual − original for changed features."""
    delta = cf - instance
    epsilon = 0.01

    changed_mask = np.abs(delta) > epsilon
    changed_names = [feature_names[i] for i in range(len(feature_names)) if changed_mask[i]]
    changed_deltas = delta[changed_mask]
    orig_vals = instance[changed_mask]
    cf_vals = cf[changed_mask]

    # Sort by absolute change descending
    order = np.argsort(np.abs(changed_deltas))[::-1]
    changed_names = [changed_names[i] for i in order]
    changed_deltas = changed_deltas[order]
    orig_vals = orig_vals[order]
    cf_vals = cf_vals[order]

    bar_colors = [COLORS["positive"] if d > 0 else COLORS["negative"]
                  for d in changed_deltas]

    # Build label annotations: "orig → cf" displayed next to each bar
    text_labels = [
        f"  {o:.1f} → {c:.1f}"
        for o, c in zip(orig_vals, cf_vals)
    ]

    fig = go.Figure(go.Bar(
        x=changed_deltas,
        y=changed_names,
        orientation="h",
        marker_color=bar_colors,
        text=text_labels,
        textposition="outside",
        textfont=dict(size=11),
        hovertemplate="%{y}: Δ = %{x:+.2f}<extra></extra>",
    ))

    fig.add_vline(x=0, line_color=COLORS["neutral"], line_width=1)

    fig.update_layout(
        title=(f"Counterfactual — What Exactly Needs to Change?<br>"
               f"<sup>To go from {original_pred:.0f} to {cf_pred:.0f} rentals, "
               f"these {len(changed_names)} features need to change (shown as Δ).</sup>"),
        xaxis_title="Change in feature value (CF − original)",
        yaxis_title="Feature",
        height=max(400, 90 + len(changed_names) * 45),
        legend=dict(orientation="h", y=-0.18),
        margin=dict(r=130),
    )
    return fig


# ── Figure 3: diverse counterfactuals — parallel coordinates ─────────────────

def plot_cf_diverse(model, instance, original_pred, X_train, feature_names) -> go.Figure:
    """Four counterfactuals via different feature constraints, shown as
    parallel coordinates alongside the original instance."""
    hr_idx = feature_names.index("hr")
    temp_idx = feature_names.index("temp")
    atemp_idx = feature_names.index("atemp")
    season_idx = feature_names.index("season")

    # CF1: only hr can change (minimise distance on all other features)
    cf1, cf1_pred = find_cf(model, X_train, instance, exclude_cols=[hr_idx])

    # CF2: only temp / atemp can change
    cf2, cf2_pred = find_cf(model, X_train, instance,
                            exclude_cols=[temp_idx, atemp_idx])

    # CF3: hr and season can change
    cf3, cf3_pred = find_cf(model, X_train, instance,
                            exclude_cols=[hr_idx, season_idx])

    # CF4: unconstrained (all features can change)
    cf4, cf4_pred = find_cf(model, X_train, instance, exclude_cols=None)

    lines = np.vstack([instance, cf1, cf2, cf3, cf4])  # (5, n_features)

    # Show top-8 features by importance for readability
    importances = model.feature_importances_
    top8_idx = np.argsort(importances)[-8:][::-1]

    dimensions = [
        dict(
            label=feature_names[feat_idx],
            values=lines[:, feat_idx].tolist(),
            range=[float(X_train[:, feat_idx].min()),
                   float(X_train[:, feat_idx].max())],
        )
        for feat_idx in top8_idx
    ]

    line_colors = [0.0, 0.25, 0.50, 0.75, 1.0]
    colorscale = [
        [0.0, COLORS["primary"]],       # original
        [0.25, COLORS["palette"][2]],   # CF1
        [0.50, COLORS["palette"][3]],   # CF2
        [0.75, COLORS["palette"][4]],   # CF3
        [1.0, COLORS["palette"][5]],    # CF4
    ]

    fig = go.Figure(go.Parcoords(
        line=dict(
            color=line_colors,
            colorscale=colorscale,
            showscale=False,
        ),
        dimensions=dimensions,
        labelangle=-25,
        labelfont=dict(size=12),
    ))

    legend_text = (
        f"<b style='color:{COLORS['primary']}'>━</b> Original (pred={original_pred:.0f})<br>"
        f"<b style='color:{COLORS['palette'][2]}'>━</b> CF1: only hr changes (pred={cf1_pred:.0f})<br>"
        f"<b style='color:{COLORS['palette'][3]}'>━</b> CF2: only temp/atemp (pred={cf2_pred:.0f})<br>"
        f"<b style='color:{COLORS['palette'][4]}'>━</b> CF3: hr + season (pred={cf3_pred:.0f})<br>"
        f"<b style='color:{COLORS['palette'][5]}'>━</b> CF4: unconstrained (pred={cf4_pred:.0f})"
    )
    fig.add_annotation(
        x=0.01, y=0.02,
        xref="paper", yref="paper",
        text=legend_text,
        showarrow=False, align="left", xanchor="left", yanchor="bottom",
        font=dict(size=11),
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor=COLORS["neutral"], borderwidth=1,
    )

    fig.update_layout(
        title=("Counterfactual — Multiple Paths to the Same Outcome<br>"
               "<sup>Each line is a different 'what-if' scenario. "
               "All CFs achieve higher rentals via different feature changes.</sup>"),
        height=500,
        margin=dict(t=100, b=120, l=80, r=80),
    )
    return fig


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    X_train, X_test, y_train, y_test, feature_names = load_regression()
    model = get_regression_model(X_train, y_train)

    inst_idx = pick_instance(model, X_test)
    instance = X_test[inst_idx]
    original_pred = float(model.predict(instance.reshape(1, -1))[0])
    print(f"Instance idx={inst_idx}, original pred={original_pred:.0f}")

    cf, cf_pred = find_cf(model, X_train, instance)
    print(f"Counterfactual pred={cf_pred:.0f}")

    print("Figure 1: CF concept scatter…")
    fig1 = plot_cf_concept(model, instance, original_pred, cf, cf_pred,
                           X_train, feature_names)
    save_figure(fig1, CHAPTER, "how_cf_concept")

    print("Figure 2: CF changes bar chart…")
    fig2 = plot_cf_changes(instance, cf, original_pred, cf_pred, feature_names)
    save_figure(fig2, CHAPTER, "how_cf_changes")

    print("Figure 3: Diverse CFs parallel coordinates…")
    fig3 = plot_cf_diverse(model, instance, original_pred, X_train, feature_names)
    save_figure(fig3, CHAPTER, "how_cf_diverse")


if __name__ == "__main__":
    main()

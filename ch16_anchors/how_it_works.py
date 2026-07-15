"""
Chapter 16: How Anchors Work
https://christophm.github.io/interpretable-ml-book/anchors.html

Conceptual figures explaining rule-based local explanations via anchors.
No anchor-exp library required — anchor regions are defined manually to
illustrate the precision/coverage trade-off.

Figures produced:
  1. how_anchor_concept.html/png          — 2D scatter with anchor rectangle
  2. how_anchor_precision_coverage.html/png — adding conditions: precision ↑, coverage ↓
  3. how_anchor_vs_lime.html/png           — anchor table vs LIME bar chart side-by-side
"""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from shared.datasets import load_classification
from shared.models import get_classification_model
from shared.theme import COLORS, save_figure

CHAPTER = "ch16_anchors"
RNG = np.random.default_rng(42)


# ── Helpers ───────────────────────────────────────────────────────────────────

def region_precision_coverage(y_pred, mask, target_class):
    """Precision = P(pred == target | mask); coverage = |mask| / |all|."""
    if not mask.any():
        return 0.0, 0.0
    precision = float((y_pred[mask] == target_class).mean())
    coverage = float(mask.sum() / len(mask))
    return precision, coverage


def build_condition_mask(X, feat_idx, threshold, direction):
    """Return boolean mask for feature > threshold or <= threshold."""
    if direction == ">":
        return X[:, feat_idx] > threshold
    return X[:, feat_idx] <= threshold


def condition_label(fname, direction, threshold):
    return f"{fname} {direction} {threshold:.3f}"


# ── Figure 1: anchor rectangle in feature space ────────────────────────────────

def plot_anchor_concept(model, X_train, X_test, feature_names, class_names) -> go.Figure:
    """Scatter of top-2 features coloured by class with the anchor region shaded."""
    importances = model.feature_importances_
    top2 = np.argsort(importances)[-2:][::-1]
    f1_idx, f2_idx = int(top2[0]), int(top2[1])
    f1_name, f2_name = feature_names[f1_idx], feature_names[f2_idx]

    X_all = np.vstack([X_train, X_test])
    y_all = model.predict(X_all)

    f1_med = float(np.median(X_all[:, f1_idx]))
    f2_med = float(np.median(X_all[:, f2_idx]))
    f1_max = float(X_all[:, f1_idx].max()) * 1.05
    f2_max = float(X_all[:, f2_idx].max()) * 1.05

    anchor_mask = (X_all[:, f1_idx] > f1_med) & (X_all[:, f2_idx] > f2_med)
    # Anchor precision: fraction inside where model predicts majority class in anchor
    anchor_pred_class = int(np.round(y_all[anchor_mask].mean()))
    precision = float((y_all[anchor_mask] == anchor_pred_class).mean())
    coverage = float(anchor_mask.sum() / len(anchor_mask))
    anchor_class_name = class_names[anchor_pred_class]

    class_colors = {0: COLORS["negative"], 1: COLORS["positive"]}

    fig = go.Figure()

    # Scatter by class
    for cls, cls_name in enumerate(class_names):
        mask = y_all == cls
        inside = mask & anchor_mask
        outside = mask & ~anchor_mask

        # Outside anchor: small, muted
        fig.add_trace(go.Scatter(
            x=X_all[outside, f1_idx], y=X_all[outside, f2_idx],
            mode="markers",
            marker=dict(color=class_colors[cls], size=5, opacity=0.25),
            name=f"{cls_name} (outside anchor)",
            legendgroup=cls_name,
            showlegend=True,
        ))

        # Inside anchor: larger, opaque
        fig.add_trace(go.Scatter(
            x=X_all[inside, f1_idx], y=X_all[inside, f2_idx],
            mode="markers",
            marker=dict(color=class_colors[cls], size=8, opacity=0.75,
                        line=dict(color="white", width=0.8)),
            name=f"{cls_name} (inside anchor)",
            legendgroup=cls_name,
            showlegend=True,
        ))

    # Anchor rectangle
    fig.add_shape(
        type="rect",
        x0=f1_med, x1=f1_max,
        y0=f2_med, y1=f2_max,
        fillcolor="rgba(37,99,235,0.10)",
        line=dict(color=COLORS["primary"], width=1.5, dash="dot"),
    )

    # Annotation inside rectangle
    fig.add_annotation(
        x=(f1_med + f1_max) / 2,
        y=(f2_med + f2_max) / 2,
        text=(f"<b>Anchor Rule</b><br>"
              f"{f1_name} > {f1_med:.2f}<br>"
              f"AND {f2_name} > {f2_med:.2f}<br>"
              f"→ predict: <b>{anchor_class_name}</b><br><br>"
              f"Precision: <b>{precision:.1%}</b><br>"
              f"Coverage: <b>{coverage:.1%}</b>"),
        showarrow=False,
        bgcolor="rgba(255,255,255,0.92)",
        bordercolor=COLORS["primary"],
        borderwidth=1,
        font=dict(size=11),
        align="center",
    )

    # Median threshold lines
    fig.add_shape(type="line", x0=f1_med, x1=f1_med,
                  y0=float(X_all[:, f2_idx].min()), y1=f2_max,
                  line=dict(color=COLORS["primary"], dash="dot", width=1))
    fig.add_shape(type="line", x0=float(X_all[:, f1_idx].min()), x1=f1_max,
                  y0=f2_med, y1=f2_med,
                  line=dict(color=COLORS["primary"], dash="dot", width=1))

    fig.update_layout(
        title=("Anchors — IF These Conditions Hold, the Prediction Almost Always Stays the Same<br>"
               "<sup>The shaded rectangle defines the anchor region. "
               "Larger dots are inside the anchor; check how cleanly one class dominates.</sup>"),
        xaxis_title=f1_name,
        yaxis_title=f2_name,
        height=520,
        legend=dict(orientation="h", y=-0.18),
    )
    return fig


# ── Figure 2: precision / coverage trade-off ─────────────────────────────────

def plot_anchor_precision_coverage(model, X_train, X_test,
                                   feature_names, class_names) -> go.Figure:
    """Adding conditions increases precision but reduces coverage."""
    importances = model.feature_importances_
    top3 = np.argsort(importances)[-3:][::-1]

    X_all = np.vstack([X_train, X_test])
    y_pred_all = model.predict(X_all)

    # Instance to explain: X_test[0]
    instance = X_test[0]
    pred_class = int(model.predict(instance.reshape(1, -1))[0])

    # Build conditions: threshold = percentile that tightens the region
    # Direction chosen so that instance is inside each expanded condition
    conditions = []
    for feat_idx in top3:
        val = float(instance[feat_idx])
        feat_median = float(np.median(X_all[:, feat_idx]))
        if val < feat_median:
            # instance is on the low side: condition "feature <= perc_70"
            thresh = float(np.percentile(X_all[:, feat_idx], 70))
            direction = "<="
        else:
            # instance is on the high side: condition "feature > perc_30"
            thresh = float(np.percentile(X_all[:, feat_idx], 30))
            direction = ">"
        conditions.append((feat_idx, thresh, direction))

    # Build cumulative masks
    precs, covs, labels = [], [], []

    # 0 conditions
    mask_0 = np.ones(len(X_all), dtype=bool)
    p0, c0 = region_precision_coverage(y_pred_all, mask_0, pred_class)
    precs.append(p0); covs.append(c0)
    labels.append("0 cond<br>(all data)")

    cumulative_mask = mask_0.copy()
    for k, (feat_idx, thresh, direction) in enumerate(conditions):
        cond_mask = build_condition_mask(X_all, feat_idx, thresh, direction)
        cumulative_mask = cumulative_mask & cond_mask
        pk, ck = region_precision_coverage(y_pred_all, cumulative_mask, pred_class)
        precs.append(pk); covs.append(ck)
        lbl = condition_label(feature_names[feat_idx], direction, thresh)
        labels.append(f"+ cond {k+1}<br>{feature_names[feat_idx][:14]}…")

    fig = go.Figure()

    # Line with markers + text
    fig.add_trace(go.Scatter(
        x=covs, y=precs,
        mode="lines+markers+text",
        text=labels,
        textposition=["bottom right", "top right", "top right", "top left"],
        marker=dict(size=12, color=COLORS["primary"],
                    line=dict(color="white", width=2)),
        line=dict(color=COLORS["primary"], width=2),
        name="Anchor (growing conditions)",
        hovertemplate="Coverage=%{x:.1%}<br>Precision=%{y:.1%}<extra></extra>",
    ))

    # Target lines
    fig.add_vline(x=0.10, line_dash="dash", line_color=COLORS["neutral"],
                  annotation_text="Min. useful coverage (10%)",
                  annotation_position="top right",
                  annotation_font_size=11)
    fig.add_hline(y=0.95, line_dash="dash", line_color=COLORS["positive"],
                  annotation_text="Target precision (95%)",
                  annotation_position="bottom right",
                  annotation_font_size=11)

    # Star for "ideal" top-left corner
    fig.add_annotation(
        x=0.08, y=0.98,
        text="★ Ideal anchor:<br>high precision,<br>acceptable coverage",
        showarrow=True, arrowhead=2, ax=50, ay=30,
        font=dict(size=11, color=COLORS["positive"]),
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor=COLORS["positive"], borderwidth=1,
    )

    fig.update_layout(
        title=("Anchors — Adding Conditions Increases Precision but Reduces Coverage<br>"
               "<sup>Each point adds one condition. Moving left means fewer instances match. "
               "Moving up means the rule is more 'pure'.</sup>"),
        xaxis=dict(title="Coverage (fraction of data where anchor applies)",
                   tickformat=".0%", range=[-0.02, 1.05]),
        yaxis=dict(title="Precision (model consistency inside anchor)",
                   tickformat=".0%", range=[0.45, 1.05]),
        height=490,
        legend=dict(orientation="h", y=-0.18),
    )
    return fig


# ── Figure 3: anchor table vs LIME bar ────────────────────────────────────────

def plot_anchor_vs_lime(model, X_train, X_test,
                        feature_names, class_names) -> go.Figure:
    """Left: formatted anchor rule table.  Right: LIME-style coefficient bar chart."""
    importances = model.feature_importances_
    top2 = np.argsort(importances)[-2:][::-1]

    X_all = np.vstack([X_train, X_test])
    y_pred_all = model.predict(X_all)

    instance = X_test[0]
    pred_class = int(model.predict(instance.reshape(1, -1))[0])
    pred_class_name = class_names[pred_class]

    # Build a 2-condition anchor (reuse same logic as Figure 2)
    conditions = []
    for feat_idx in top2:
        val = float(instance[feat_idx])
        feat_median = float(np.median(X_all[:, feat_idx]))
        if val < feat_median:
            thresh = float(np.percentile(X_all[:, feat_idx], 70))
            direction = "<="
        else:
            thresh = float(np.percentile(X_all[:, feat_idx], 30))
            direction = ">"
        conditions.append((feat_idx, thresh, direction))

    mask = np.ones(len(X_all), dtype=bool)
    for feat_idx, thresh, direction in conditions:
        mask = mask & build_condition_mask(X_all, feat_idx, thresh, direction)
    precision, coverage = region_precision_coverage(y_pred_all, mask, pred_class)

    cond_strs = [
        condition_label(feature_names[fi], d, t)
        for fi, t, d in conditions
    ]

    # ── LIME local Ridge ──
    sigma = X_train.std(axis=0) * 0.30
    X_perturb = RNG.normal(loc=instance, scale=sigma, size=(300, instance.shape[0]))
    for j in range(instance.shape[0]):
        X_perturb[:, j] = np.clip(X_perturb[:, j],
                                   X_train[:, j].min(), X_train[:, j].max())

    proba_perturb = model.predict_proba(X_perturb)[:, pred_class]

    std_scale = X_train.std(axis=0) + 1e-8
    kernel_width = 0.75 * np.sqrt(instance.shape[0])
    dists = np.linalg.norm((X_perturb - instance) / std_scale, axis=1)
    weights = np.exp(-dists ** 2 / (2 * kernel_width ** 2))

    scaler_lime = StandardScaler()
    X_std = scaler_lime.fit_transform(X_perturb)
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_std, proba_perturb, sample_weight=weights)

    coefs = ridge.coef_
    top8_idx = np.argsort(np.abs(coefs))[-8:][::-1]
    top8_names = [feature_names[i] for i in top8_idx]
    top8_coefs = coefs[top8_idx]

    # ── Subplots ──
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "table"}, {"type": "xy"}]],
        subplot_titles=[
            "Anchor Explanation (rule-based)",
            "LIME Explanation (weight-based)",
        ],
        column_widths=[0.45, 0.55],
        horizontal_spacing=0.08,
    )

    # Left: anchor table
    row_labels = ["IF", "AND", "THEN:", "", "Precision:", "Coverage:"]
    row_values = [
        cond_strs[0],
        cond_strs[1],
        f"predict '{pred_class_name}'",
        "",
        f"{precision:.1%}",
        f"{coverage:.1%}",
    ]
    row_fills = [
        "rgba(37,99,235,0.10)", "rgba(37,99,235,0.10)",
        "rgba(5,150,105,0.12)", "white",
        "rgba(255,255,255,0)", "rgba(255,255,255,0)",
    ]

    fig.add_trace(go.Table(
        header=dict(
            values=["<b>Rule component</b>", "<b>Value</b>"],
            fill_color=COLORS["primary"],
            font=dict(color="white", size=12),
            align="left",
            height=30,
        ),
        cells=dict(
            values=[row_labels, row_values],
            fill_color=[list(row_fills), list(row_fills)],
            align=["left", "left"],
            font=dict(size=12, color="#1F2937"),
            height=28,
        ),
    ), row=1, col=1)

    # Right: LIME coefficient bar chart
    bar_colors = [
        COLORS["positive"] if c > 0 else COLORS["negative"]
        for c in top8_coefs
    ]
    # Abbreviate long feature names
    short_names = [n[:22] + "…" if len(n) > 22 else n for n in top8_names]

    fig.add_trace(go.Bar(
        x=top8_coefs,
        y=short_names,
        orientation="h",
        marker_color=bar_colors,
        name="LIME weight",
        hovertemplate="%{y}: %{x:+.4f}<extra></extra>",
    ), row=1, col=2)

    fig.update_xaxes(title_text="Standardised coefficient (local Ridge)",
                     row=1, col=2)

    fig.add_annotation(
        x=0.98, y=0.03,
        xref="paper", yref="paper",
        text=(
            "<b>Anchors</b>: IF–THEN rule — discrete, binary.<br>"
            "Works even when conditions change slightly.<br><br>"
            "<b>LIME</b>: weighted coefficients — continuous.<br>"
            "Shows direction and magnitude of each feature."
        ),
        showarrow=False,
        align="right", xanchor="right", yanchor="bottom",
        font=dict(size=11),
        bgcolor="rgba(255,255,255,0.90)",
        bordercolor=COLORS["neutral"], borderwidth=1,
    )

    fig.update_layout(
        title=("Anchors vs LIME — Rule-Based vs Weight-Based Local Explanation<br>"
               "<sup>Same instance, two explanation styles. "
               "Anchors give IF-THEN rules; LIME gives feature weights.</sup>"),
        height=490,
        showlegend=False,
    )
    return fig


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    X_train, X_test, y_train, y_test, feature_names, class_names = load_classification()
    model = get_classification_model(X_train, y_train)

    inst = X_test[0]
    pred = class_names[int(model.predict(inst.reshape(1, -1))[0])]
    print(f"Instance 0 predicted class: {pred}")

    print("Figure 1: Anchor concept scatter…")
    fig1 = plot_anchor_concept(model, X_train, X_test, feature_names, class_names)
    save_figure(fig1, CHAPTER, "how_anchor_concept")

    print("Figure 2: Precision / coverage trade-off…")
    fig2 = plot_anchor_precision_coverage(model, X_train, X_test,
                                          feature_names, class_names)
    save_figure(fig2, CHAPTER, "how_anchor_precision_coverage")

    print("Figure 3: Anchor vs LIME…")
    fig3 = plot_anchor_vs_lime(model, X_train, X_test, feature_names, class_names)
    save_figure(fig3, CHAPTER, "how_anchor_vs_lime")


if __name__ == "__main__":
    main()

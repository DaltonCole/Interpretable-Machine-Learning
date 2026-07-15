"""
Chapter 24: How LOFO Importance Works
https://christophm.github.io/interpretable-ml-book/lofo.html

Conceptual figures explaining the method — separate from model-output figures in main.py.
LOFO is implemented manually (no lofo-importance library required).

Figures produced:
  1. how_lofo_concept.html/png      — 3-step pipeline diagram + actual vs predicted comparison
  2. how_lofo_all_features.html/png — LOFO importance for all 12 features
  3. how_lofo_vs_perm.html/png      — scatter comparing LOFO and permutation importance
"""
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score

from shared.datasets import load_regression
from shared.models import get_regression_model
from shared.theme import COLORS, save_figure

CHAPTER = "ch24_lofo"
RNG = np.random.default_rng(42)
FAST_ESTIMATORS = 30  # smaller model for the per-feature retrain loop


def lofo_importance(X_train, X_test, y_train, y_test, feature_names, base_r2):
    """Manual LOFO: retrain without each feature, return dict of importance values."""
    importances = {}
    for i, name in enumerate(feature_names):
        cols = [j for j in range(X_train.shape[1]) if j != i]
        m = RandomForestRegressor(n_estimators=FAST_ESTIMATORS, random_state=42, n_jobs=-1)
        m.fit(X_train[:, cols], y_train)
        r2_without = r2_score(y_test, m.predict(X_test[:, cols]))
        importances[name] = base_r2 - r2_without
    return importances


# ── Figure 1: Concept diagram + prediction comparison ────────────────────────

def plot_lofo_concept(model, X_train, X_test, y_train, y_test, feature_names) -> go.Figure:
    base_r2 = r2_score(y_test, model.predict(X_test))

    hr_idx = feature_names.index("hr")
    cols_no_hr = [j for j in range(X_train.shape[1]) if j != hr_idx]
    model_no_hr = RandomForestRegressor(n_estimators=FAST_ESTIMATORS, random_state=42, n_jobs=-1)
    model_no_hr.fit(X_train[:, cols_no_hr], y_train)
    r2_no_hr = r2_score(y_test, model_no_hr.predict(X_test[:, cols_no_hr]))
    lofo_hr = base_r2 - r2_no_hr

    c_pos = COLORS["positive"]
    c_neg = COLORS["negative"]

    # ---- Layout: 3-step diagram (top) + scatter comparison (bottom) ----------
    fig = go.Figure()

    # Box 1: train on all
    fig.add_shape(type="rect", x0=0.02, x1=0.30, y0=0.62, y1=0.98,
                  xref="paper", yref="paper",
                  fillcolor="#EFF6FF", line=dict(color=COLORS["primary"], width=2))
    fig.add_annotation(
        xref="paper", yref="paper", x=0.16, y=0.865,
        text=(
            "<b>Step 1</b><br>"
            "Train on<br>ALL features<br>"
            f"<span style='color:{c_pos}'>R² = {base_r2:.3f}</span>"
        ),
        showarrow=False, align="center", font=dict(size=12),
    )

    # Arrow 1 → 2
    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.395, y=0.80, ax=-60, ay=0,
        text="", showarrow=True, arrowhead=2, arrowcolor=COLORS["neutral"],
    )

    # Box 2: remove hr and retrain
    fig.add_shape(type="rect", x0=0.41, x1=0.60, y0=0.62, y1=0.98,
                  xref="paper", yref="paper",
                  fillcolor="#FFF7ED", line=dict(color=COLORS["accent"], width=2))
    fig.add_annotation(
        xref="paper", yref="paper", x=0.505, y=0.865,
        text=(
            "<b>Step 2</b><br>"
            "Remove 'hr',<br>retrain<br>"
            f"<span style='color:{c_neg}'>R² = {r2_no_hr:.3f}</span>"
        ),
        showarrow=False, align="center", font=dict(size=12),
    )

    # Arrow 2 → 3
    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.685, y=0.80, ax=-52, ay=0,
        text="", showarrow=True, arrowhead=2, arrowcolor=COLORS["neutral"],
    )

    # Box 3: LOFO importance
    fig.add_shape(type="rect", x0=0.70, x1=0.98, y0=0.62, y1=0.98,
                  xref="paper", yref="paper",
                  fillcolor="#F0FDF4", line=dict(color=COLORS["positive"], width=2))
    fig.add_annotation(
        xref="paper", yref="paper", x=0.84, y=0.865,
        text=(
            "<b>Step 3</b><br>"
            f"LOFO importance =<br>"
            f"{base_r2:.3f} − {r2_no_hr:.3f}<br>"
            f"<b><span style='color:{c_pos}'>= {lofo_hr:.3f}</span></b>"
        ),
        showarrow=False, align="center", font=dict(size=12),
    )

    # Scatter: actual vs predicted WITH hr (blue) and WITHOUT hr (red dashed)
    idx_samp = RNG.choice(len(X_test), size=300, replace=False)
    y_samp = y_test[idx_samp]
    pred_with = model.predict(X_test[idx_samp])
    pred_without = model_no_hr.predict(X_test[idx_samp, :][:, cols_no_hr])

    diag = [float(y_test.min()), float(y_test.max())]

    fig.add_trace(go.Scatter(
        x=y_samp, y=pred_with, mode="markers",
        marker=dict(color=COLORS["primary"], size=4, opacity=0.5),
        name=f"With hr (R²={base_r2:.3f})",
        xaxis="x", yaxis="y",
        hovertemplate="Actual=%{x:.0f}<br>Predicted=%{y:.0f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=y_samp, y=pred_without, mode="markers",
        marker=dict(color=COLORS["negative"], size=4, opacity=0.5),
        name=f"Without hr (R²={r2_no_hr:.3f})",
        xaxis="x", yaxis="y",
        hovertemplate="Actual=%{x:.0f}<br>Predicted=%{y:.0f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=diag, y=diag, mode="lines",
        line=dict(dash="dash", color=COLORS["neutral"], width=1.5),
        name="Perfect fit",
        xaxis="x", yaxis="y",
    ))

    fig.update_layout(
        title=(
            "LOFO — Leave One Feature Out and Measure the Performance Drop<br>"
            "<sup>Retrain without each feature; the R² drop is the LOFO importance</sup>"
        ),
        xaxis=dict(title="Actual rentals", domain=[0, 1], anchor="y"),
        yaxis=dict(title="Predicted rentals", domain=[0, 0.55], anchor="x"),
        height=540,
        legend=dict(orientation="h", y=-0.12),
    )
    return fig


# ── Figure 2: All-feature LOFO importance ────────────────────────────────────

def plot_lofo_all_features(model, X_train, X_test, y_train, y_test, feature_names) -> go.Figure:
    base_r2 = r2_score(y_test, model.predict(X_test))
    importances = lofo_importance(X_train, X_test, y_train, y_test, feature_names, base_r2)

    names = list(importances.keys())
    vals = list(importances.values())
    order = np.argsort(vals)

    sorted_names = [names[i] for i in order]
    sorted_vals = [vals[i] for i in order]
    sorted_colors = [
        COLORS["positive"] if v > 0 else COLORS["negative"]
        for v in sorted_vals
    ]

    top_name = names[int(np.argmax(vals))]
    top_val = max(vals)

    fig = go.Figure(go.Bar(
        x=sorted_vals,
        y=sorted_names,
        orientation="h",
        marker_color=sorted_colors,
        text=[f"{v:+.3f}" for v in sorted_vals],
        textposition="outside",
        hovertemplate="%{y}: LOFO = %{x:+.4f}<extra></extra>",
    ))

    fig.add_vline(x=0, line_color=COLORS["neutral"], line_width=1.5, line_dash="dash")

    fig.add_annotation(
        x=top_val * 0.5,
        y=len(sorted_names) - 1,
        text=f"Removing '{top_name}' drops R²<br>by {top_val:.3f} — most important",
        showarrow=True,
        arrowhead=2,
        ax=80, ay=-30,
        font=dict(size=12, color=COLORS["positive"]),
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor=COLORS["positive"],
        borderwidth=1,
    )

    fig.update_layout(
        title=(
            "LOFO — Ranking Features by Their Contribution to Model Performance<br>"
            "<sup>Positive = feature helped; negative = removing it actually improved R²</sup>"
        ),
        xaxis_title="LOFO importance (base R² − R² without feature)",
        height=520,
        legend=dict(orientation="h", y=-0.18),
    )
    return fig


# ── Figure 3: LOFO vs permutation scatter ────────────────────────────────────

def plot_lofo_vs_perm(model, X_train, X_test, y_train, y_test, feature_names) -> go.Figure:
    base_r2 = r2_score(y_test, model.predict(X_test))
    lofo_vals = lofo_importance(X_train, X_test, y_train, y_test, feature_names, base_r2)

    perm_result = permutation_importance(
        model, X_test, y_test, n_repeats=10, random_state=42, scoring="r2"
    )
    perm_vals = {name: float(perm_result.importances_mean[i])
                 for i, name in enumerate(feature_names)}

    lofo_arr = np.array([lofo_vals[n] for n in feature_names])
    perm_arr = np.array([perm_vals[n] for n in feature_names])

    # Diagonal reference line
    lo = min(perm_arr.min(), lofo_arr.min())
    hi = max(perm_arr.max(), lofo_arr.max())
    pad = (hi - lo) * 0.1

    # Color by magnitude (sum of both metrics, larger = more important)
    magnitude = np.abs(lofo_arr) + np.abs(perm_arr)

    # Detect outliers: large residual from diagonal
    diff = lofo_arr - perm_arr
    outlier_threshold = float(np.std(diff)) * 1.5
    is_outlier = np.abs(diff) > outlier_threshold

    fig = go.Figure()

    # Diagonal
    fig.add_trace(go.Scatter(
        x=[lo - pad, hi + pad], y=[lo - pad, hi + pad],
        mode="lines",
        line=dict(dash="dash", color=COLORS["neutral"], width=1.5),
        name="Perfect agreement",
        showlegend=True,
    ))

    # Points
    fig.add_trace(go.Scatter(
        x=perm_arr,
        y=lofo_arr,
        mode="markers+text",
        text=feature_names,
        textposition="top center",
        marker=dict(
            color=magnitude,
            colorscale="Viridis",
            size=10,
            showscale=False,
        ),
        name="Feature",
        hovertemplate=(
            "<b>%{text}</b><br>"
            "Permutation: %{x:.4f}<br>"
            "LOFO: %{y:.4f}<extra></extra>"
        ),
    ))

    # Annotate the biggest outlier
    if is_outlier.any():
        worst_idx = int(np.argmax(np.abs(diff)))
        worst_name = feature_names[worst_idx]
        fig.add_annotation(
            x=perm_arr[worst_idx],
            y=lofo_arr[worst_idx],
            text=f"'{worst_name}' disagrees:<br>LOFO vs perm differ by {diff[worst_idx]:+.3f}",
            showarrow=True,
            arrowhead=2,
            ax=60, ay=-40,
            font=dict(size=11, color=COLORS["accent"]),
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor=COLORS["accent"],
            borderwidth=1,
        )

    fig.update_layout(
        title=(
            "LOFO vs Permutation — Two Approaches to 'What If We Lost This Feature?'<br>"
            "<sup>Points on the diagonal agree; outliers reveal where retraining vs "
            "permuting give different answers</sup>"
        ),
        xaxis_title="Permutation importance (ΔR²)",
        yaxis_title="LOFO importance (ΔR²)",
        height=510,
        legend=dict(orientation="h", y=-0.18),
    )
    return fig


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    X_train, X_test, y_train, y_test, feature_names = load_regression()
    model = get_regression_model(X_train, y_train)

    print("Figure 1: LOFO concept diagram…")
    fig1 = plot_lofo_concept(model, X_train, X_test, y_train, y_test, feature_names)
    save_figure(fig1, CHAPTER, "how_lofo_concept")

    print("Figure 2: LOFO for all features (12 retrains)…")
    fig2 = plot_lofo_all_features(model, X_train, X_test, y_train, y_test, feature_names)
    save_figure(fig2, CHAPTER, "how_lofo_all_features")

    print("Figure 3: LOFO vs permutation scatter…")
    fig3 = plot_lofo_vs_perm(model, X_train, X_test, y_train, y_test, feature_names)
    save_figure(fig3, CHAPTER, "how_lofo_vs_perm")


if __name__ == "__main__":
    main()

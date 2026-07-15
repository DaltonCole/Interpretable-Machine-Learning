"""
Chapter 25: How Global Surrogate Models Work
https://christophm.github.io/interpretable-ml-book/global.html

Figures produced:
  1. how_surrogate_concept.html/png   — black-box vs surrogate fidelity scatter
  2. how_surrogate_depth.html/png     — fidelity vs depth tradeoff curve
  3. how_surrogate_rules.html/png     — rules extracted from the surrogate tree
"""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score

from shared.datasets import load_regression
from shared.models import get_regression_model
from shared.theme import COLORS, save_figure

CHAPTER = "ch25_surrogate_models"
RNG = np.random.default_rng(42)


# ── Figure 1: Black-box vs surrogate fidelity ──────────────────────────────

def plot_surrogate_concept(blackbox, X_train, X_test, y_test, feature_names):
    bb_preds_train = blackbox.predict(X_train)
    bb_preds_test  = blackbox.predict(X_test)

    surrogate = DecisionTreeRegressor(max_depth=4, random_state=42)
    surrogate.fit(X_train, bb_preds_train)
    surr_preds = surrogate.predict(X_test)

    fidelity_r2 = r2_score(bb_preds_test, surr_preds)
    actual_r2   = r2_score(y_test, bb_preds_test)

    residuals_bb   = np.abs(y_test - bb_preds_test)
    discrepancy    = np.abs(surr_preds - bb_preds_test)
    cmax_disc      = float(np.percentile(discrepancy, 95))

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[
            f"Black-box Fit (R²={actual_r2:.3f})",
            f"Surrogate Fidelity (R²={fidelity_r2:.3f})",
        ],
        horizontal_spacing=0.12,
    )

    # Left: actual vs black-box predicted
    cmax_bb = float(np.percentile(residuals_bb, 95))
    fig.add_trace(go.Scatter(
        x=bb_preds_test, y=y_test,
        mode="markers",
        marker=dict(color=residuals_bb, colorscale="RdYlGn_r",
                    cmin=0, cmax=cmax_bb, size=4, opacity=0.5, showscale=False),
        name="Test instances",
        hovertemplate="BB pred: %{x:.0f}<br>Actual: %{y:.0f}<extra></extra>",
    ), row=1, col=1)
    lim1 = [min(bb_preds_test.min(), y_test.min()),
            max(bb_preds_test.max(), y_test.max())]
    fig.add_trace(go.Scatter(x=lim1, y=lim1, mode="lines",
        line=dict(dash="dash", color=COLORS["neutral"], width=1.5),
        name="Perfect fit", showlegend=True), row=1, col=1)
    fig.add_annotation(row=1, col=1, xref="x domain", yref="y domain",
        x=0.03, y=0.97, xanchor="left", yanchor="top",
        text="● green = small error<br>● red = large error",
        showarrow=False, font=dict(size=11),
        bgcolor="rgba(255,255,255,0.85)", bordercolor="#D1D5DB", borderwidth=1)

    # Right: black-box vs surrogate
    fig.add_trace(go.Scatter(
        x=bb_preds_test, y=surr_preds,
        mode="markers",
        marker=dict(color=discrepancy, colorscale="RdYlGn_r",
                    cmin=0, cmax=cmax_disc, size=4, opacity=0.5, showscale=False),
        name="BB vs Surrogate",
        hovertemplate="BB: %{x:.0f}<br>Surrogate: %{y:.0f}<extra></extra>",
    ), row=1, col=2)
    lim2 = [min(bb_preds_test.min(), surr_preds.min()),
            max(bb_preds_test.max(), surr_preds.max())]
    fig.add_trace(go.Scatter(x=lim2, y=lim2, mode="lines",
        line=dict(dash="dash", color=COLORS["neutral"], width=1.5),
        name="Perfect fidelity", showlegend=True), row=1, col=2)
    fig.add_annotation(row=1, col=2, xref="x2 domain", yref="y2 domain",
        x=0.03, y=0.97, xanchor="left", yanchor="top",
        text="● green = high fidelity<br>● red = surrogate disagrees",
        showarrow=False, font=dict(size=11),
        bgcolor="rgba(255,255,255,0.85)", bordercolor="#D1D5DB", borderwidth=1)

    fig.update_xaxes(title_text="Black-box predicted rentals", row=1, col=1)
    fig.update_yaxes(title_text="Actual rentals", row=1, col=1)
    fig.update_xaxes(title_text="Black-box predicted rentals", row=1, col=2)
    fig.update_yaxes(title_text="Surrogate predicted rentals", row=1, col=2)

    fig.update_layout(
        title=("Global Surrogate — Train an Interpretable Model to Mimic a Black Box<br>"
               "<sup>Left: how well the black box fits actuals. "
               "Right: how closely a depth-4 decision tree mimics the black box.</sup>"),
        height=500, legend=dict(orientation="h", y=-0.18),
    )
    return fig


# ── Figure 2: Fidelity vs depth ────────────────────────────────────────────

def plot_surrogate_depth(blackbox, X_train, X_test):
    bb_preds_train = blackbox.predict(X_train)
    bb_preds_test  = blackbox.predict(X_test)

    depths = list(range(1, 13))
    fidelity, actual_r2 = [], []
    for d in depths:
        surr = DecisionTreeRegressor(max_depth=d, random_state=42)
        surr.fit(X_train, bb_preds_train)
        sp = surr.predict(X_test)
        fidelity.append(r2_score(bb_preds_test, sp))
        actual_r2.append(r2_score(bb_preds_test, sp))  # same as fidelity here

    # Also compute actual-label R² for surrogate
    actual_label_r2 = []
    y_train_arr = None  # loaded in main and passed as global; compute from bb_preds instead
    for d in depths:
        surr = DecisionTreeRegressor(max_depth=d, random_state=42)
        surr.fit(X_train, bb_preds_train)
        sp = surr.predict(X_test)
        actual_label_r2.append(r2_score(bb_preds_test, sp))  # fidelity

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=depths, y=fidelity, mode="lines+markers",
        line=dict(color=COLORS["primary"], width=2.5),
        marker=dict(size=8),
        name="Fidelity to black-box (R²)",
        hovertemplate="Depth %{x}: fidelity=%{y:.3f}<extra></extra>",
    ))

    # Shade interpretable zone (depth ≤ 4)
    max_y = 1.0
    fig.add_shape(type="rect", x0=0.5, x1=4.5, y0=0, y1=max_y,
        fillcolor="rgba(5,150,105,0.07)", line_width=0)
    fig.add_annotation(x=2.5, y=0.97,
        text="Interpretable<br>(depth ≤ 4)", showarrow=False,
        font=dict(size=12, color=COLORS["positive"]),
        bgcolor="rgba(255,255,255,0.85)", bordercolor=COLORS["positive"], borderwidth=1)

    # Best fidelity point
    best_d = int(np.argmax(fidelity)) + 1
    fig.add_vline(x=4, line_dash="dot", line_color=COLORS["positive"], line_width=2)
    fig.add_annotation(x=4, y=fidelity[3] - 0.04,
        text=f"Typical choice: depth=4<br>fidelity={fidelity[3]:.3f}",
        showarrow=True, arrowhead=2, ax=60, ay=-40,
        font=dict(size=12, color=COLORS["positive"]),
        bgcolor="rgba(255,255,255,0.9)", bordercolor=COLORS["positive"], borderwidth=1)
    fig.add_annotation(x=best_d, y=fidelity[best_d-1] + 0.02,
        text=f"Peak fidelity at depth={best_d}",
        showarrow=True, arrowhead=2, ax=0, ay=-35,
        font=dict(size=12, color=COLORS["primary"]),
        bgcolor="rgba(255,255,255,0.9)", bordercolor=COLORS["primary"], borderwidth=1)

    fig.update_layout(
        title=("Global Surrogate — Deeper Trees Mimic the Black Box Better but Become Opaque<br>"
               "<sup>Fidelity R² measures how closely the surrogate reproduces black-box predictions "
               "(not actual labels). Deeper = better mimicry, worse interpretability.</sup>"),
        xaxis=dict(title="Surrogate tree max_depth", tickmode="linear", dtick=1),
        yaxis=dict(title="Fidelity R² (surrogate vs black-box)", range=[0, 1.02]),
        height=500, legend=dict(orientation="h", y=-0.18),
    )
    return fig


# ── Figure 3: Rules extracted from depth-4 surrogate ─────────────────────

def _extract_paths(tree, feature_names, max_paths=8):
    """Walk a fitted DecisionTreeRegressor and return leaf paths as strings."""
    left  = tree.tree_.children_left
    right = tree.tree_.children_right
    feat  = tree.tree_.feature
    thres = tree.tree_.threshold
    value = tree.tree_.value
    n_node_samples = tree.tree_.n_node_samples
    n_total = n_node_samples[0]

    paths = []

    def recurse(node, conditions):
        if left[node] == right[node]:  # leaf
            pred = float(value[node][0][0])
            n    = int(n_node_samples[node])
            frac = n / n_total
            paths.append({
                "rule": " AND\n  ".join(conditions) if conditions else "(no conditions — root)",
                "prediction": pred,
                "n_samples": n,
                "coverage": frac,
            })
            return
        fname = feature_names[feat[node]]
        thr   = thres[node]
        recurse(left[node],  conditions + [f"{fname} ≤ {thr:.2f}"])
        recurse(right[node], conditions + [f"{fname} > {thr:.2f}"])

    recurse(0, [])
    paths.sort(key=lambda p: p["n_samples"], reverse=True)
    return paths[:max_paths]


def plot_surrogate_rules(blackbox, X_train, X_test, y_test, feature_names):
    bb_preds_train = blackbox.predict(X_train)
    surrogate = DecisionTreeRegressor(max_depth=4, random_state=42)
    surrogate.fit(X_train, bb_preds_train)

    paths = _extract_paths(surrogate, feature_names, max_paths=8)
    total_coverage = sum(p["coverage"] for p in paths)

    rules_text  = ["IF " + p["rule"].replace("\n", "<br>   ") for p in paths]
    preds_text  = [f"{p['prediction']:.0f}" for p in paths]
    counts_text = [f"{p['n_samples']:,}" for p in paths]
    cov_text    = [f"{p['coverage']:.1%}" for p in paths]

    row_colors = [["#EFF6FF" if i % 2 == 0 else "white" for i in range(len(paths))]] * 4
    header_vals = ["Rule (IF conditions)", "Predicted<br>rentals", "# instances", "Coverage"]

    fig = go.Figure(go.Table(
        columnwidth=[400, 80, 80, 70],
        header=dict(
            values=[f"<b>{h}</b>" for h in header_vals],
            fill_color=COLORS["primary"],
            font=dict(color="white", size=13),
            align=["left", "center", "center", "center"],
            height=36,
        ),
        cells=dict(
            values=[rules_text, preds_text, counts_text, cov_text],
            fill_color=row_colors,
            font=dict(size=11),
            align=["left", "center", "center", "center"],
            height=40,
        ),
    ))

    fig.add_annotation(
        x=0.5, y=-0.06, xref="paper", yref="paper",
        text=(f"These {len(paths)} paths cover {total_coverage:.0%} of training instances. "
              "Each path is one route through the depth-4 surrogate tree."),
        showarrow=False, font=dict(size=12, color=COLORS["neutral"]),
    )

    fig.update_layout(
        title=("Global Surrogate — A Depth-4 Tree Distils the Black Box into Human-Readable Rules<br>"
               "<sup>Each row is one leaf path. The surrogate tree approximates the random forest "
               "using only simple IF-THEN rules.</sup>"),
        height=560, margin=dict(t=80, b=70),
    )
    return fig


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    X_train, X_test, y_train, y_test, feature_names = load_regression()
    blackbox = get_regression_model(X_train, y_train)

    print("Figure 1: Surrogate concept (fidelity scatter)…")
    fig1 = plot_surrogate_concept(blackbox, X_train, X_test, y_test, feature_names)
    save_figure(fig1, CHAPTER, "how_surrogate_concept")

    print("Figure 2: Fidelity vs depth tradeoff…")
    fig2 = plot_surrogate_depth(blackbox, X_train, X_test)
    save_figure(fig2, CHAPTER, "how_surrogate_depth")

    print("Figure 3: Extracted rules from surrogate tree…")
    fig3 = plot_surrogate_rules(blackbox, X_train, X_test, y_test, feature_names)
    save_figure(fig3, CHAPTER, "how_surrogate_rules")


if __name__ == "__main__":
    main()

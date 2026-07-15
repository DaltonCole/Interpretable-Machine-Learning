"""
Chapter 9: How Decision Trees Work
https://christophm.github.io/interpretable-ml-book/tree.html

Conceptual figures explaining the method — separate from the model-output
figures in main.py.

Figures produced:
  1. how_tree_split.html/png      — 2D feature space partitioned by a shallow tree
  2. how_tree_path.html/png       — decision path for one test instance through the tree
  3. how_tree_complexity.html/png — bias-variance tradeoff across tree depths
"""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor, _tree

from shared.datasets import load_regression
from shared.theme import COLORS, save_figure

CHAPTER = "ch09_decision_tree"
RNG = np.random.default_rng(42)


def _collect_splits(tree_obj, feature_names, node=0, bounds=None):
    """Recursively collect all (split_line_info) and (leaf_bounds, leaf_pred) from a tree."""
    if bounds is None:
        bounds = {}

    t = tree_obj.tree_
    if t.feature[node] == _tree.TREE_UNDEFINED:
        pred = float(t.value[node][0][0])
        return [], [(bounds.copy(), pred)]

    feat_name = feature_names[t.feature[node]]
    threshold = float(t.threshold[node])

    splits = [(feat_name, threshold, bounds.copy())]

    left_bounds  = bounds.copy()
    right_bounds = bounds.copy()
    lo, hi = bounds.get(feat_name, (None, None))
    left_bounds[feat_name]  = (lo, threshold)
    right_bounds[feat_name] = (threshold, hi)

    l_splits, l_leaves = _collect_splits(tree_obj, feature_names, t.children_left[node],  left_bounds)
    r_splits, r_leaves = _collect_splits(tree_obj, feature_names, t.children_right[node], right_bounds)

    return splits + l_splits + r_splits, l_leaves + r_leaves


# ── Figure 1: 2D split visualisation ─────────────────────────────────────────

def plot_tree_split(X_train, X_test, y_train, y_test, feature_names) -> go.Figure:
    """2D scatter of hr vs temp colored by rental count, with tree decision boundaries."""
    hr_idx   = feature_names.index("hr")
    temp_idx = feature_names.index("temp")
    feat2    = ["hr", "temp"]

    X2_train = X_train[:, [hr_idx, temp_idx]]

    tree = DecisionTreeRegressor(max_depth=3, random_state=42)
    tree.fit(X2_train, y_train)

    splits, leaves = _collect_splits(tree, feat2, bounds={"hr": (None, None), "temp": (None, None)})

    hr_min,   hr_max   = float(X_train[:, hr_idx].min()),   float(X_train[:, hr_idx].max())
    temp_min, temp_max = float(X_train[:, temp_idx].min()), float(X_train[:, temp_idx].max())

    samp = RNG.choice(len(X_test), size=min(600, len(X_test)), replace=False)

    fig = go.Figure()

    # Scatter colored by actual rental count
    fig.add_trace(go.Scatter(
        x=X_test[samp, hr_idx],
        y=X_test[samp, temp_idx],
        mode="markers",
        marker=dict(
            color=y_test[samp],
            colorscale="Viridis",
            size=6,
            opacity=0.65,
            colorbar=dict(title="Rentals", thickness=12, len=0.7),
        ),
        name="Test instances",
        hovertemplate="hr=%{x:.0f}, temp=%{y:.2f}<br>Rentals=%{marker.color:.0f}<extra></extra>",
    ))

    split_colors = [COLORS["accent"], COLORS["secondary"], COLORS["neutral"],
                    "#D97706", "#0891B2", "#9333EA"]

    for i, (feat_name, threshold, bounds) in enumerate(splits):
        col = split_colors[i % len(split_colors)]
        if feat_name == "hr":
            lo_y = bounds.get("temp", (None, None))[0]
            hi_y = bounds.get("temp", (None, None))[1]
            y0 = lo_y if lo_y is not None else temp_min
            y1 = hi_y if hi_y is not None else temp_max
            fig.add_trace(go.Scatter(
                x=[threshold, threshold], y=[y0, y1],
                mode="lines",
                line=dict(color=col, width=2, dash="solid"),
                name=f"hr ≤ {threshold:.1f}",
                hoverinfo="skip",
            ))
        elif feat_name == "temp":
            lo_x = bounds.get("hr", (None, None))[0]
            hi_x = bounds.get("hr", (None, None))[1]
            x0 = lo_x if lo_x is not None else hr_min
            x1 = hi_x if hi_x is not None else hr_max
            fig.add_trace(go.Scatter(
                x=[x0, x1], y=[threshold, threshold],
                mode="lines",
                line=dict(color=col, width=2, dash="solid"),
                name=f"temp ≤ {threshold:.2f}",
                hoverinfo="skip",
            ))

    # Label leaf regions with predicted value
    for bounds, pred in leaves:
        lo_hr   = bounds.get("hr",   (None, None))[0];   lo_hr   = lo_hr   if lo_hr   is not None else hr_min
        hi_hr   = bounds.get("hr",   (None, None))[1];   hi_hr   = hi_hr   if hi_hr   is not None else hr_max
        lo_temp = bounds.get("temp", (None, None))[0];   lo_temp = lo_temp if lo_temp is not None else temp_min
        hi_temp = bounds.get("temp", (None, None))[1];   hi_temp = hi_temp if hi_temp is not None else temp_max
        cx = (lo_hr + hi_hr) / 2
        cy = (lo_temp + hi_temp) / 2
        fig.add_annotation(
            x=cx, y=cy,
            text=f"ŷ={pred:.0f}",
            showarrow=False,
            font=dict(size=10, color="#111827"),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="#D1D5DB", borderwidth=1,
        )

    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.01, y=0.01,
        text="Each rectangle is a leaf — all instances in it get the same prediction",
        showarrow=False, align="left",
        font=dict(size=11, color=COLORS["neutral"]),
        bgcolor="rgba(255,255,255,0.85)", bordercolor="#D1D5DB", borderwidth=1,
    )

    fig.update_layout(
        title=("Decision Tree — Recursive Binary Splits Partition the Feature Space<br>"
               "<sup>A depth-3 tree on 'hr' and 'temp': each split divides the 2D space "
               "into regions; every point in a region gets the same leaf prediction.</sup>"),
        xaxis=dict(title="Hour of day (hr)", tickmode="linear", tick0=0, dtick=2),
        yaxis_title="Temperature (temp, normalised)",
        height=520,
        legend=dict(orientation="h", y=-0.18),
    )
    return fig


# ── Figure 2: Decision path for one instance ─────────────────────────────────

def plot_tree_path(X_train, X_test, y_train, y_test, feature_names) -> go.Figure:
    """Horizontal flow table tracing one test instance through a depth-4 tree."""
    tree = DecisionTreeRegressor(max_depth=4, random_state=42)
    tree.fit(X_train, y_train)

    inst   = X_test[0]
    actual = y_test[0]

    t = tree.tree_
    node = 0
    path_nodes = []
    while True:
        feat_idx  = t.feature[node]
        threshold = float(t.threshold[node])
        pred_here = float(t.value[node][0][0])
        n_samples = int(t.n_node_samples[node])
        is_leaf   = (feat_idx == _tree.TREE_UNDEFINED)

        path_nodes.append({
            "depth":     len(path_nodes),
            "feat":      feature_names[feat_idx] if not is_leaf else None,
            "threshold": threshold if not is_leaf else None,
            "feat_val":  float(inst[feat_idx]) if not is_leaf else None,
            "pred":      pred_here,
            "n":         n_samples,
            "is_leaf":   is_leaf,
        })

        if is_leaf:
            break
        node = t.children_left[node] if inst[feat_idx] <= threshold else t.children_right[node]

    depth_labels  = [f"Depth {p['depth']}" for p in path_nodes]
    condition_col = []
    decision_col  = []
    pred_col      = [f"{p['pred']:.0f}" for p in path_nodes]
    samples_col   = [str(p["n"])         for p in path_nodes]

    for p in path_nodes:
        if p["is_leaf"]:
            condition_col.append("(leaf)")
            decision_col.append(f"PREDICT  {p['pred']:.0f} rentals")
        else:
            condition_col.append(f"{p['feat']} ≤ {p['threshold']:.2f}?")
            goes = "YES →" if p["feat_val"] <= p["threshold"] else "NO →"
            decision_col.append(f"{p['feat']} = {p['feat_val']:.2f}   {goes}")

    row_fill = ["#DCFCE7" if p["is_leaf"] else "#EFF6FF" for p in path_nodes]

    fig = go.Figure(data=[go.Table(
        columnwidth=[55, 170, 200, 80, 80],
        header=dict(
            values=["<b>Depth</b>", "<b>Condition</b>", "<b>Feature value / Decision</b>",
                    "<b>Node avg</b>", "<b>Samples</b>"],
            fill_color=COLORS["primary"],
            font=dict(color="white", size=13),
            align="left",
            height=36,
        ),
        cells=dict(
            values=[depth_labels, condition_col, decision_col, pred_col, samples_col],
            fill_color=[row_fill] * 5,
            align="left",
            font=dict(size=12),
            height=34,
        ),
    )])

    final_pred = path_nodes[-1]["pred"]
    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.5, y=-0.1,
        text=(f"Final prediction: <b>{final_pred:.0f}</b> rentals  |  "
              f"Actual: <b>{actual:.0f}</b> rentals  |  "
              f"Error: {abs(final_pred - actual):.0f}"),
        showarrow=False,
        font=dict(size=13, color=COLORS["primary"]),
        bgcolor="rgba(255,255,255,0.9)", bordercolor=COLORS["primary"], borderwidth=1,
    )

    n_rows = len(path_nodes)
    fig.update_layout(
        title=("Decision Tree — Trace the Path to Any Prediction<br>"
               "<sup>Test instance #0 routed left (≤) or right (>) at each node; "
               "the highlighted leaf gives the final prediction.</sup>"),
        height=max(480, 100 + n_rows * 52),
        margin=dict(t=90, b=90),
    )
    return fig


# ── Figure 3: Bias-variance tradeoff across depths ───────────────────────────

def plot_tree_complexity(X_train, X_test, y_train, y_test) -> go.Figure:
    """Train and test R² across max_depth 1–12; shaded under- and overfitting regions."""
    depths   = list(range(1, 13))
    train_r2 = []
    test_r2  = []

    for d in depths:
        tree = DecisionTreeRegressor(max_depth=d, random_state=42)
        tree.fit(X_train, y_train)
        train_r2.append(r2_score(y_train, tree.predict(X_train)))
        test_r2.append(r2_score(y_test,  tree.predict(X_test)))

    train_r2 = np.array(train_r2)
    test_r2  = np.array(test_r2)
    opt_idx  = int(np.argmax(test_r2))
    opt_d    = depths[opt_idx]

    fig = go.Figure()

    # Underfitting region
    fig.add_trace(go.Scatter(
        x=[depths[0] - 0.4, opt_d, opt_d, depths[0] - 0.4],
        y=[0, 0, 1.02, 1.02],
        fill="toself", fillcolor="rgba(217,119,6,0.08)",
        line=dict(color="rgba(0,0,0,0)"),
        name="Underfitting region", hoverinfo="skip",
    ))

    # Overfitting region
    fig.add_trace(go.Scatter(
        x=[opt_d, depths[-1] + 0.4, depths[-1] + 0.4, opt_d],
        y=[0, 0, 1.02, 1.02],
        fill="toself", fillcolor="rgba(220,38,38,0.08)",
        line=dict(color="rgba(0,0,0,0)"),
        name="Overfitting region", hoverinfo="skip",
    ))

    fig.add_trace(go.Scatter(
        x=depths, y=train_r2,
        mode="lines+markers",
        line=dict(color=COLORS["primary"], width=2.5),
        marker=dict(size=8),
        name="Train R²",
        hovertemplate="depth=%{x}<br>Train R²=%{y:.3f}<extra></extra>",
    ))

    fig.add_trace(go.Scatter(
        x=depths, y=test_r2,
        mode="lines+markers",
        line=dict(color=COLORS["positive"], width=2.5),
        marker=dict(size=8),
        name="Test R²",
        hovertemplate="depth=%{x}<br>Test R²=%{y:.3f}<extra></extra>",
    ))

    fig.add_vline(x=opt_d, line_dash="dash", line_color=COLORS["accent"], line_width=2)
    fig.add_annotation(
        x=opt_d, y=float(test_r2[opt_idx]) + 0.04,
        text=f"Optimal depth = {opt_d}<br>Test R² = {test_r2[opt_idx]:.3f}",
        showarrow=True, arrowhead=2, ax=60, ay=-35,
        font=dict(size=12, color=COLORS["accent"]),
        bgcolor="rgba(255,255,255,0.9)", bordercolor=COLORS["accent"], borderwidth=1,
    )

    mid_under = (depths[0] + opt_d) / 2
    mid_over  = (opt_d + depths[-1]) / 2
    fig.add_annotation(
        x=mid_under, y=0.18, text="Underfitting<br>(too simple)",
        showarrow=False, font=dict(size=12, color="#D97706"),
        bgcolor="rgba(255,255,255,0.85)",
    )
    fig.add_annotation(
        x=mid_over, y=0.18, text="Overfitting<br>(memorises noise)",
        showarrow=False, font=dict(size=12, color=COLORS["negative"]),
        bgcolor="rgba(255,255,255,0.85)",
    )

    fig.add_annotation(
        xref="paper", yref="paper", x=0.01, y=0.01,
        text="Sweet spot: deep enough to capture patterns, not so deep it memorises noise",
        showarrow=False, align="left",
        font=dict(size=11, color=COLORS["neutral"]),
        bgcolor="rgba(255,255,255,0.85)", bordercolor="#D1D5DB", borderwidth=1,
    )

    fig.update_layout(
        title=("Decision Tree — Depth Controls the Bias-Variance Tradeoff<br>"
               "<sup>Train R² always rises with depth; test R² peaks then levels off "
               "as the tree memorises training noise.</sup>"),
        xaxis=dict(title="max_depth", tickmode="linear", tick0=1, dtick=1),
        yaxis=dict(title="R²", range=[0, 1.05]),
        height=520,
        legend=dict(orientation="h", y=-0.18),
    )
    return fig


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    X_train, X_test, y_train, y_test, feature_names = load_regression()

    print("Figure 1: 2D split visualisation…")
    fig1 = plot_tree_split(X_train, X_test, y_train, y_test, feature_names)
    save_figure(fig1, CHAPTER, "how_tree_split")

    print("Figure 2: Decision path for one instance…")
    fig2 = plot_tree_path(X_train, X_test, y_train, y_test, feature_names)
    save_figure(fig2, CHAPTER, "how_tree_path")

    print("Figure 3: Complexity / bias-variance tradeoff…")
    fig3 = plot_tree_complexity(X_train, X_test, y_train, y_test)
    save_figure(fig3, CHAPTER, "how_tree_complexity")


if __name__ == "__main__":
    main()

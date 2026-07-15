"""
Chapter 10: How Decision Rules Work
https://christophm.github.io/interpretable-ml-book/rules.html

Conceptual figures explaining the method — separate from the model-output
figures in main.py.

Figures produced:
  1. how_rules_concept.html/png  — 2D scatter with rule rectangles overlaid
  2. how_rules_list.html/png     — ordered decision list with instance tracing
  3. how_rules_tradeoff.html/png — coverage vs confidence bubble chart per rule
"""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, _tree

from shared.datasets import load_classification
from shared.theme import COLORS, save_figure

CHAPTER = "ch10_decision_rules"
RNG = np.random.default_rng(42)


def extract_rules(tree, feature_names, class_names, X_train, y_train):
    """Extract leaf-level IF-THEN rules from a fitted DecisionTreeClassifier."""
    t = tree.tree_
    rules = []

    def recurse(node, conditions):
        if t.feature[node] != _tree.TREE_UNDEFINED:
            feat      = feature_names[t.feature[node]]
            thresh    = t.threshold[node]
            recurse(t.children_left[node],  conditions + [(feat, "<=", thresh)])
            recurse(t.children_right[node], conditions + [(feat, ">",  thresh)])
        else:
            values    = t.value[node][0]
            total     = values.sum()
            pred_cls  = int(np.argmax(values))
            confidence = float(values[pred_cls] / total)
            # Evaluate coverage on training data
            mask = np.ones(len(X_train), dtype=bool)
            for feat, op, thresh in conditions:
                fidx = list(feature_names).index(feat)
                if op == "<=":
                    mask &= X_train[:, fidx] <= thresh
                else:
                    mask &= X_train[:, fidx] > thresh
            coverage = int(mask.sum())
            rules.append({
                "conditions": conditions,
                "rule_str":   _format_rule(conditions),
                "prediction": class_names[pred_cls],
                "pred_cls":   pred_cls,
                "confidence": confidence,
                "coverage":   coverage,
            })

    recurse(0, [])
    return rules


def _format_rule(conditions, max_len=70):
    parts = [f"{feat} {op} {thresh:.2f}" for feat, op, thresh in conditions]
    s = " AND ".join(parts) if parts else "True"
    return s if len(s) <= max_len else s[:max_len] + "…"


# ── Figure 1: Rule rectangles on 2D scatter ──────────────────────────────────

def plot_rules_concept(X_train, y_train, feature_names, class_names) -> go.Figure:
    """2D scatter with semi-transparent rule rectangles showing class regions."""
    # Find the 2 most important features by RF
    rf = RandomForestClassifier(n_estimators=50, random_state=42)
    rf.fit(X_train, y_train)
    top2 = np.argsort(rf.feature_importances_)[::-1][:2]
    f0_idx, f1_idx = int(top2[0]), int(top2[1])
    f0_name, f1_name = feature_names[f0_idx], feature_names[f1_idx]

    # Train a shallow tree on just these 2 features
    X2 = X_train[:, [f0_idx, f1_idx]]
    feat2 = [f0_name, f1_name]
    tree = DecisionTreeClassifier(max_depth=3, random_state=42)
    tree.fit(X2, y_train)

    rules = extract_rules(tree, feat2, class_names, X2, y_train)
    # Top 3 by confidence × coverage (balanced)
    for r in rules:
        r["score"] = r["confidence"] * np.log1p(r["coverage"])
    top_rules = sorted(rules, key=lambda r: r["score"], reverse=True)[:3]

    f0_min, f0_max = float(X_train[:, f0_idx].min()), float(X_train[:, f0_idx].max())
    f1_min, f1_max = float(X_train[:, f1_idx].min()), float(X_train[:, f1_idx].max())

    fig = go.Figure()

    # Scatter by class
    for cls_idx, cls_name, col in [(0, class_names[0], COLORS["negative"]),
                                    (1, class_names[1], COLORS["positive"])]:
        mask = y_train == cls_idx
        fig.add_trace(go.Scatter(
            x=X_train[mask, f0_idx],
            y=X_train[mask, f1_idx],
            mode="markers",
            marker=dict(color=col, size=5, opacity=0.45),
            name=cls_name,
        ))

    # Draw rule rectangles
    rect_colors = ["rgba(37,99,235,0.12)", "rgba(124,58,237,0.12)", "rgba(5,150,105,0.12)"]
    rect_border = [COLORS["primary"], COLORS["secondary"], COLORS["positive"]]

    for i, rule in enumerate(top_rules):
        # Compute bounding box from conditions
        x0, x1, y0, y1 = f0_min, f0_max, f1_min, f1_max
        for feat, op, thresh in rule["conditions"]:
            if feat == f0_name:
                if op == "<=":
                    x1 = min(x1, thresh)
                else:
                    x0 = max(x0, thresh)
            elif feat == f1_name:
                if op == "<=":
                    y1 = min(y1, thresh)
                else:
                    y0 = max(y0, thresh)

        fig.add_shape(
            type="rect",
            x0=x0, y0=y0, x1=x1, y1=y1,
            fillcolor=rect_colors[i],
            line=dict(color=rect_border[i], width=2, dash="dash"),
        )

        # Label the rectangle
        pred_col = COLORS["negative"] if rule["pred_cls"] == 0 else COLORS["positive"]
        short_rule = rule["rule_str"][:55] + ("…" if len(rule["rule_str"]) > 55 else "")
        fig.add_annotation(
            x=(x0 + x1) / 2, y=y1,
            text=f"IF {short_rule}<br>THEN {rule['prediction']} (conf={rule['confidence']:.0%})",
            showarrow=True, arrowhead=2,
            ax=0, ay=-50,
            font=dict(size=10, color=rect_border[i]),
            bgcolor="rgba(255,255,255,0.92)",
            bordercolor=rect_border[i], borderwidth=1,
        )

    fig.update_layout(
        title=("Decision Rules — IF-THEN Conditions That Cover a Region of Feature Space<br>"
               "<sup>Each rectangle corresponds to one rule; instances inside get the same "
               "class prediction with the stated confidence.</sup>"),
        xaxis_title=f0_name,
        yaxis_title=f1_name,
        height=520,
        legend=dict(orientation="h", y=-0.18),
    )
    return fig


# ── Figure 2: Decision list with instance trace ───────────────────────────────

def plot_rules_list(X_train, X_test, y_train, y_test, feature_names, class_names) -> go.Figure:
    """Ordered decision list showing an instance flowing down until a rule fires."""
    tree = DecisionTreeClassifier(max_depth=3, random_state=42)
    tree.fit(X_train, y_train)
    rules = extract_rules(tree, feature_names, class_names, X_train, y_train)

    # Sort by confidence descending — simulates a decision list priority
    rules_sorted = sorted(rules, key=lambda r: r["confidence"], reverse=True)[:6]

    # Pick a test instance and find which rule fires first
    inst     = X_test[0]
    inst_cls = int(y_test[0])

    def rule_fires(rule, x):
        for feat, op, thresh in rule["conditions"]:
            fidx = list(feature_names).index(feat)
            if op == "<=" and x[fidx] > thresh:
                return False
            if op == ">" and x[fidx] <= thresh:
                return False
        return True

    fired_idx = None
    for i, rule in enumerate(rules_sorted):
        if rule_fires(rule, inst):
            fired_idx = i
            break

    # Build table rows
    rule_labels  = [f"Rule {i+1}" for i in range(len(rules_sorted))]
    condition_col = [r["rule_str"][:65] + ("…" if len(r["rule_str"]) > 65 else "")
                     for r in rules_sorted]
    pred_col      = [r["prediction"]         for r in rules_sorted]
    conf_col      = [f"{r['confidence']:.0%}" for r in rules_sorted]
    cov_col       = [str(r["coverage"])       for r in rules_sorted]

    row_fill = []
    for i in range(len(rules_sorted)):
        if i == fired_idx:
            row_fill.append("#DCFCE7")   # green: matched
        elif fired_idx is not None and i < fired_idx:
            row_fill.append("#FEF3C7")   # amber: tested but didn't fire
        else:
            row_fill.append("white")     # not reached

    fig = go.Figure(data=[go.Table(
        columnwidth=[55, 260, 80, 60, 60],
        header=dict(
            values=["<b>Priority</b>", "<b>Conditions</b>", "<b>Prediction</b>",
                    "<b>Confidence</b>", "<b>Coverage</b>"],
            fill_color=COLORS["primary"],
            font=dict(color="white", size=13),
            align="left",
            height=36,
        ),
        cells=dict(
            values=[rule_labels, condition_col, pred_col, conf_col, cov_col],
            fill_color=[row_fill] * 5,
            align="left",
            font=dict(size=11),
            height=34,
        ),
    )])

    if fired_idx is not None:
        matched = rules_sorted[fired_idx]
        status = ("correctly" if matched["pred_cls"] == inst_cls else "incorrectly")
        fig.add_annotation(
            xref="paper", yref="paper",
            x=0.5, y=-0.08,
            text=(f"Test instance #0 fires Rule {fired_idx + 1} → "
                  f"predicted <b>{matched['prediction']}</b> ({status}) &nbsp;|&nbsp; "
                  f"Actual: <b>{class_names[inst_cls]}</b>"),
            showarrow=False,
            font=dict(size=13, color=COLORS["primary"]),
            bgcolor="rgba(255,255,255,0.9)", bordercolor=COLORS["primary"], borderwidth=1,
        )
    else:
        fig.add_annotation(
            xref="paper", yref="paper", x=0.5, y=-0.08,
            text="Instance did not match any rule (default: majority class)",
            showarrow=False, font=dict(size=12, color=COLORS["neutral"]),
        )

    fig.update_layout(
        title=("Decision List — Rules Are Applied in Priority Order Until One Fires<br>"
               "<sup>Green = matched rule; amber = tested but conditions not met; "
               "white = not reached. First match wins.</sup>"),
        height=max(480, 120 + len(rules_sorted) * 50),
        margin=dict(t=90, b=90),
    )
    return fig


# ── Figure 3: Coverage vs Confidence bubble chart ────────────────────────────

def plot_rules_tradeoff(X_train, y_train, feature_names, class_names) -> go.Figure:
    """Bubble chart: each rule plotted by coverage (x) vs confidence (y), size=coverage."""
    tree = DecisionTreeClassifier(max_depth=4, random_state=42)
    tree.fit(X_train, y_train)
    rules = extract_rules(tree, feature_names, class_names, X_train, y_train)
    rules = [r for r in rules if r["coverage"] > 0]

    coverages   = np.array([r["coverage"]   for r in rules])
    confidences = np.array([r["confidence"] for r in rules])
    pred_cls    = np.array([r["pred_cls"]   for r in rules])
    rule_strs   = [r["rule_str"]            for r in rules]

    # Normalize bubble size
    size_norm = 8 + 32 * (coverages - coverages.min()) / (coverages.max() - coverages.min() + 1)

    fig = go.Figure()

    for cls_idx, cls_name, col in [(0, class_names[0], COLORS["negative"]),
                                    (1, class_names[1], COLORS["positive"])]:
        mask = pred_cls == cls_idx
        fig.add_trace(go.Scatter(
            x=coverages[mask],
            y=confidences[mask],
            mode="markers",
            marker=dict(
                color=col,
                size=size_norm[mask],
                opacity=0.65,
                line=dict(color="white", width=1),
            ),
            name=f"Predicts: {cls_name}",
            text=[rule_strs[i] for i in range(len(rules)) if pred_cls[i] == cls_idx],
            hovertemplate="Coverage: %{x}<br>Confidence: %{y:.1%}<br>%{text}<extra></extra>",
        ))

    # Reference lines
    fig.add_hline(y=0.5, line_dash="dash", line_color=COLORS["neutral"], line_width=1.5)
    fig.add_annotation(
        xref="paper", yref="y",
        x=0.01, y=0.52,
        text="Random chance (conf = 0.5)", showarrow=False,
        font=dict(size=11, color=COLORS["neutral"]),
    )

    q25 = float(np.percentile(coverages, 25))
    fig.add_vline(x=q25, line_dash="dot", line_color=COLORS["neutral"], line_width=1)
    fig.add_annotation(
        xref="x", yref="paper",
        x=q25, y=0.02,
        text="Low coverage threshold",
        showarrow=False, textangle=-90,
        font=dict(size=10, color=COLORS["neutral"]),
    )

    # Corner annotations
    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.02, y=0.98,
        text="Narrow rules =<br>high confidence, low coverage",
        showarrow=False, align="left", xanchor="left", yanchor="top",
        font=dict(size=11, color=COLORS["primary"]),
        bgcolor="rgba(255,255,255,0.85)", bordercolor=COLORS["primary"], borderwidth=1,
    )
    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.98, y=0.08,
        text="Broad rules =<br>lower confidence, high coverage",
        showarrow=False, align="right", xanchor="right", yanchor="bottom",
        font=dict(size=11, color=COLORS["secondary"]),
        bgcolor="rgba(255,255,255,0.85)", bordercolor=COLORS["secondary"], borderwidth=1,
    )

    fig.update_layout(
        title=("Decision Rules — Coverage vs Confidence Tradeoff<br>"
               "<sup>Bubble size = coverage. Precise rules cover fewer instances with higher accuracy; "
               "general rules cover more instances but trade off precision.</sup>"),
        xaxis_title="Coverage (# training instances covered by rule)",
        yaxis=dict(title="Confidence (accuracy on covered instances)", tickformat=".0%", range=[0, 1.05]),
        height=520,
        legend=dict(orientation="h", y=-0.18),
    )
    return fig


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    X_train, X_test, y_train, y_test, feature_names, class_names = load_classification()

    print("Figure 1: Rule rectangles on 2D scatter…")
    fig1 = plot_rules_concept(X_train, y_train, feature_names, class_names)
    save_figure(fig1, CHAPTER, "how_rules_concept")

    print("Figure 2: Decision list with instance trace…")
    fig2 = plot_rules_list(X_train, X_test, y_train, y_test, feature_names, class_names)
    save_figure(fig2, CHAPTER, "how_rules_list")

    print("Figure 3: Coverage vs confidence tradeoff…")
    fig3 = plot_rules_tradeoff(X_train, y_train, feature_names, class_names)
    save_figure(fig3, CHAPTER, "how_rules_tradeoff")


if __name__ == "__main__":
    main()

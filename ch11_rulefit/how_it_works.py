"""
Chapter 11: How RuleFit Works
https://christophm.github.io/interpretable-ml-book/rulefit.html

Conceptual figures explaining the method — separate from the model-output
figures in main.py.

Figures produced:
  1. how_rulefit_pipeline.html/png        — visual 3-stage pipeline diagram
  2. how_rulefit_rule_activation.html/png — distribution of rentals when top rule fires vs not
  3. how_rulefit_decompose.html/png       — prediction waterfall: which rules fired?
"""
import numpy as np
import plotly.graph_objects as go
from imodels import RuleFitRegressor

from shared.datasets import load_regression
from shared.theme import COLORS, save_figure

CHAPTER = "ch11_rulefit"
RNG = np.random.default_rng(42)


def _truncate(text, max_len=60):
    return text if len(text) <= max_len else text[:max_len] + "…"


def _evaluate_rule(rule_str, x, feature_names):
    """Evaluate a rule string against a single feature vector x.

    Rules look like: 'hr > 8.5 & hr <= 17.5' or 'temp > 0.42'.
    Returns True if the rule fires, False otherwise, None if rule is a linear term.
    """
    if rule_str is None or not isinstance(rule_str, str):
        return None
    # Linear terms in imodels use feature names directly (no comparator)
    if "<=" not in rule_str and ">" not in rule_str and "<" not in rule_str:
        return None  # linear term, not a boolean rule

    result = True
    # imodels may use ' & ' or ' and ' as separator depending on version
    sep = " and " if " and " in rule_str else "&"
    for part in rule_str.split(sep):
        part = part.strip()
        # Try >= first (before >)
        if ">=" in part:
            feat, val = part.split(">=")
            feat = feat.strip()
            val  = float(val.strip())
            if feat in feature_names:
                idx = feature_names.index(feat)
                result = result and (x[idx] >= val)
        elif "<=" in part:
            feat, val = part.split("<=")
            feat = feat.strip()
            val  = float(val.strip())
            if feat in feature_names:
                idx = feature_names.index(feat)
                result = result and (x[idx] <= val)
        elif ">" in part:
            feat, val = part.split(">")
            feat = feat.strip()
            val  = float(val.strip())
            if feat in feature_names:
                idx = feature_names.index(feat)
                result = result and (x[idx] > val)
        elif "<" in part:
            feat, val = part.split("<")
            feat = feat.strip()
            val  = float(val.strip())
            if feat in feature_names:
                idx = feature_names.index(feat)
                result = result and (x[idx] < val)
    return result


def _apply_rules_batch(rule_str, X, feature_names):
    """Apply a rule to every row of X; return bool array."""
    n = len(X)
    out = np.zeros(n, dtype=bool)
    for i in range(n):
        fired = _evaluate_rule(rule_str, X[i], feature_names)
        out[i] = bool(fired) if fired is not None else False
    return out


# ── Figure 1: Pipeline diagram ────────────────────────────────────────────────

def plot_rulefit_pipeline(model, feature_names) -> go.Figure:
    """Visual 3-stage pipeline: Trees → Rules → Lasso. With rule-type bar chart below."""
    rules_df = model._get_rules()
    rules_df = rules_df[rules_df["coef"] != 0].copy()

    n_rules  = int((rules_df["type"] == "rule").sum())
    n_linear = int((rules_df["type"] == "linear").sum())
    total    = n_rules + n_linear

    fig = go.Figure()

    # ── Box positions (y=0.65 for pipeline row) ──────────────────────────────
    boxes = [
        dict(x0=0.02, y0=0.45, x1=0.27, y1=0.92,
             label="Stage 1",
             title="Tree Ensemble",
             lines=["Gradient-boosted", "trees → leaf paths", f"~{total} paths total"]),
        dict(x0=0.37, y0=0.45, x1=0.62, y1=0.92,
             label="Stage 2",
             title="Extract Rules",
             lines=["Each path becomes", "IF … THEN …", f"{n_rules} binary rules"]),
        dict(x0=0.72, y0=0.45, x1=0.97, y1=0.92,
             label="Stage 3",
             title="Lasso Selection",
             lines=["Sparse weights on", "rules + features", f"{n_linear} terms kept"]),
    ]

    box_colors  = [COLORS["primary"], COLORS["secondary"], COLORS["accent"]]
    arrow_color = COLORS["neutral"]

    for box, col in zip(boxes, box_colors):
        # Box background
        fig.add_shape(
            type="rect",
            x0=box["x0"], y0=box["y0"], x1=box["x1"], y1=box["y1"],
            xref="paper", yref="paper",
            fillcolor=f"rgba(37,99,235,0.07)" if col == COLORS["primary"]
                      else f"rgba(124,58,237,0.07)" if col == COLORS["secondary"]
                      else "rgba(219,39,119,0.07)",
            line=dict(color=col, width=2),
        )
        cx = (box["x0"] + box["x1"]) / 2
        # Stage label
        fig.add_annotation(
            xref="paper", yref="paper",
            x=cx, y=box["y1"] - 0.03,
            text=f"<b>{box['label']}</b>",
            showarrow=False, font=dict(size=11, color=col),
        )
        # Title
        fig.add_annotation(
            xref="paper", yref="paper",
            x=cx, y=box["y1"] - 0.09,
            text=f"<b>{box['title']}</b>",
            showarrow=False, font=dict(size=11, color="#111827"),
        )
        # Detail lines
        for j, line in enumerate(box["lines"]):
            fig.add_annotation(
                xref="paper", yref="paper",
                x=cx, y=box["y1"] - 0.17 - j * 0.10,
                text=line,
                showarrow=False, font=dict(size=11, color="#374151"),
            )

    # Arrows between boxes (line shapes + arrowhead annotations)
    for i in range(len(boxes) - 1):
        x_start = boxes[i]["x1"]
        x_end   = boxes[i+1]["x0"]
        y_mid   = (boxes[i]["y0"] + boxes[i]["y1"]) / 2
        # Shaft
        fig.add_shape(
            type="line",
            x0=x_start, y0=y_mid, x1=x_end - 0.01, y1=y_mid,
            xref="paper", yref="paper",
            line=dict(color=arrow_color, width=2),
        )
        # Arrowhead: annotation at destination with a tiny pixel offset so it renders
        fig.add_annotation(
            xref="paper", yref="paper",
            x=x_end, y=y_mid,
            ax=-8, ay=0,          # pixel offset (axref/ayref default = pixel)
            showarrow=True, arrowhead=2, arrowwidth=2,
            arrowcolor=arrow_color,
            text="",
        )

    # Combined label
    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.5, y=0.38,
        text="<b>ŷ = β₀ + Σ aⱼ · ruleⱼ(x) + Σ βₖ · xₖ</b>",
        showarrow=False, font=dict(size=14, color=COLORS["primary"]),
        bgcolor="rgba(255,255,255,0.9)", bordercolor=COLORS["primary"], borderwidth=1,
    )

    # ── Bar chart: rule type distribution ────────────────────────────────────
    fig.add_trace(go.Bar(
        x=["Boolean rules", "Linear terms"],
        y=[n_rules, n_linear],
        marker_color=[COLORS["primary"], COLORS["accent"]],
        text=[str(n_rules), str(n_linear)],
        textposition="outside",
        hovertemplate="%{x}: %{y} terms<extra></extra>",
        name="Non-zero model terms",
        showlegend=False,
    ))

    fig.update_xaxes(domain=[0.1, 0.9])
    fig.update_yaxes(
        title_text="# non-zero Lasso terms",
        domain=[0, 0.30],
    )

    fig.update_layout(
        title=("RuleFit — Three Stages: Trees → Rules → Lasso<br>"
               "<sup>Boolean rules and original linear features are combined "
               "via Lasso, yielding a sparse, interpretable model.</sup>"),
        height=540,
        margin=dict(t=90, b=50),
        legend=dict(orientation="h", y=-0.12),
    )
    return fig


# ── Figure 2: Rule activation distribution ────────────────────────────────────

def plot_rulefit_rule_activation(model, X_train, y_train, feature_names) -> go.Figure:
    """Violin plots of rental counts when the most important rule fires vs not."""
    rules_df = model._get_rules()
    rules_df = rules_df[rules_df["coef"] != 0].copy()
    rules_df["importance"] = np.abs(rules_df["coef"]) * rules_df["support"]

    # Filter to actual boolean rules (not linear terms)
    rule_rows = rules_df[rules_df["type"] == "rule"].sort_values("importance", ascending=False)

    # Find the top rule that actually fires on some training instances
    top_rule_str = None
    fires_mask   = None
    for _, row in rule_rows.iterrows():
        mask = _apply_rules_batch(row["rule"], X_train, feature_names)
        if mask.sum() > 10 and (~mask).sum() > 10:
            top_rule_str = row["rule"]
            fires_mask   = mask
            break

    if top_rule_str is None:
        # Fallback: use the top rule regardless
        top_rule_str = rule_rows.iloc[0]["rule"]
        fires_mask   = _apply_rules_batch(top_rule_str, X_train, feature_names)

    y_fires    = y_train[fires_mask]
    y_no_fires = y_train[~fires_mask]

    mean_fires    = float(y_fires.mean())    if len(y_fires) > 0    else 0.0
    mean_no_fires = float(y_no_fires.mean()) if len(y_no_fires) > 0 else 0.0

    fig = go.Figure()

    fig.add_trace(go.Violin(
        x=["Rule fires"] * len(y_fires),
        y=y_fires,
        name="Rule fires (=1)",
        fillcolor=f"rgba(37,99,235,0.35)",
        line_color=COLORS["primary"],
        box_visible=True,
        meanline_visible=True,
        points="outliers",
        hovertemplate="Rentals=%{y:.0f}<extra></extra>",
    ))

    fig.add_trace(go.Violin(
        x=["Rule does not fire"] * len(y_no_fires),
        y=y_no_fires,
        name="Rule doesn't fire (=0)",
        fillcolor=f"rgba(219,39,119,0.25)",
        line_color=COLORS["accent"],
        box_visible=True,
        meanline_visible=True,
        points="outliers",
        hovertemplate="Rentals=%{y:.0f}<extra></extra>",
    ))

    display_rule = _truncate(top_rule_str, 80)
    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.5, y=1.0,
        text=f"Rule: <i>{display_rule}</i>",
        showarrow=False, align="center",
        font=dict(size=11, color=COLORS["neutral"]),
        bgcolor="rgba(255,255,255,0.9)", bordercolor="#D1D5DB", borderwidth=1,
    )

    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.01, y=0.01,
        text=(f"When rule fires: avg = {mean_fires:.0f} rentals  |  "
              f"When it doesn't: avg = {mean_no_fires:.0f} rentals  |  "
              f"Difference: {abs(mean_fires - mean_no_fires):.0f}"),
        showarrow=False, align="left",
        font=dict(size=12, color=COLORS["primary"]),
        bgcolor="rgba(255,255,255,0.9)", bordercolor=COLORS["primary"], borderwidth=1,
    )

    fig.update_layout(
        title=("RuleFit — A Rule Is a Binary Feature That Fires or Doesn't<br>"
               "<sup>When the top-importance rule is satisfied, "
               "the rental distribution shifts substantially — that is the rule's signal.</sup>"),
        xaxis_title="Rule activation",
        yaxis_title="Bike rentals (cnt)",
        violingap=0.3,
        height=490,
        legend=dict(orientation="h", y=-0.18),
    )
    return fig


# ── Figure 3: Prediction waterfall — which rules fired? ──────────────────────

def plot_rulefit_decompose(model, X_test, y_test, feature_names) -> go.Figure:
    """Waterfall: intercept → fired rules → linear terms → final prediction."""
    rules_df = model._get_rules()
    rules_df = rules_df[rules_df["coef"] != 0].copy()
    rules_df["importance"] = np.abs(rules_df["coef"]) * rules_df["support"]

    # Pick the test instance closest to median prediction
    preds_all = model.predict(X_test)
    inst_idx  = int(np.argmin(np.abs(preds_all - np.median(preds_all))))
    instance  = X_test[inst_idx]
    prediction = float(preds_all[inst_idx])
    actual     = float(y_test[inst_idx])

    # Separate rules and linear terms
    rule_rows   = rules_df[rules_df["type"] == "rule"]
    linear_rows = rules_df[rules_df["type"] == "linear"]

    # Compute contributions
    contributions = []
    for _, row in rule_rows.iterrows():
        fires = _evaluate_rule(row["rule"], instance, feature_names)
        if fires is None:
            fires = False
        contrib = float(row["coef"]) * (1.0 if fires else 0.0)
        if abs(contrib) > 0:
            contributions.append({
                "label": _truncate(row["rule"], 40),
                "contrib": contrib,
                "detail": f"Rule fires: {'YES' if fires else 'NO'}\ncoef = {row['coef']:.2f}",
            })

    for _, row in linear_rows.iterrows():
        feat = row["rule"]
        if feat in feature_names:
            fidx   = feature_names.index(feat)
            val    = float(instance[fidx])
            contrib = float(row["coef"]) * val
            if abs(contrib) > 0.5:
                contributions.append({
                    "label": f"linear: {feat}",
                    "contrib": contrib,
                    "detail": f"{feat} = {val:.2f}\ncoef = {row['coef']:.2f}",
                })

    # Sort by absolute contribution, top 10
    contributions.sort(key=lambda c: abs(c["contrib"]), reverse=True)
    contributions = contributions[:10]

    # Model intercept (mean of training predictions)
    intercept = float(model.predict(X_test).mean())  # approximate; imodels doesn't expose it cleanly
    # Recompute from prediction and contributions
    contrib_sum = sum(c["contrib"] for c in contributions)
    intercept   = prediction - contrib_sum

    # Build waterfall
    running   = intercept
    x_labels  = ["Intercept"]
    bar_vals   = [intercept]
    bar_bases  = [0]
    bar_colors = [COLORS["neutral"]]
    hover_txts = [f"Baseline (intercept) = {intercept:.1f}"]

    for c in contributions:
        x_labels.append(c["label"])
        bar_vals.append(c["contrib"])
        bar_bases.append(running)
        bar_colors.append(COLORS["positive"] if c["contrib"] >= 0 else COLORS["negative"])
        hover_txts.append(c["detail"] + f"\nContribution: {c['contrib']:+.1f}")
        running += c["contrib"]

    x_labels.append("Prediction")
    bar_vals.append(prediction)
    bar_bases.append(0)
    bar_colors.append(COLORS["primary"])
    hover_txts.append(f"Final prediction: {prediction:.0f} rentals\nActual: {actual:.0f}")

    fig = go.Figure()

    for lbl, val, base, col, htxt in zip(x_labels[:-1], bar_vals[:-1], bar_bases[:-1],
                                          bar_colors[:-1], hover_txts[:-1]):
        tpl = f"{val:.0f}" if lbl == "Intercept" else f"{val:+.0f}"
        fig.add_trace(go.Bar(
            x=[lbl], y=[val], base=[base],
            marker_color=col, showlegend=False,
            hovertemplate=htxt.replace("\n", "<br>") + "<extra></extra>",
            texttemplate=tpl, textposition="outside",
        ))

    fig.add_trace(go.Bar(
        x=["Prediction"], y=[prediction], base=[0],
        marker_color=COLORS["primary"], showlegend=False,
        hovertemplate=hover_txts[-1].replace("\n", "<br>") + "<extra></extra>",
        texttemplate=f"{prediction:.0f}", textposition="outside",
    ))

    fig.add_hline(y=intercept, line_dash="dash", line_color=COLORS["neutral"], line_width=1)

    fig.add_annotation(
        x="Prediction", y=prediction + 30,
        text=f"Predicted: <b>{prediction:.0f}</b><br>Actual: <b>{actual:.0f}</b>",
        showarrow=True, arrowhead=2, ax=70, ay=-40,
        font=dict(size=12, color=COLORS["primary"]),
        bgcolor="rgba(255,255,255,0.9)", bordercolor=COLORS["primary"], borderwidth=1,
    )

    fig.add_trace(go.Bar(x=[None], y=[None], marker_color=COLORS["positive"], name="Increases prediction"))
    fig.add_trace(go.Bar(x=[None], y=[None], marker_color=COLORS["negative"], name="Decreases prediction"))
    fig.add_trace(go.Bar(x=[None], y=[None], marker_color=COLORS["neutral"],  name="Intercept (baseline)"))

    fig.update_layout(
        title=("RuleFit — Prediction Breakdown: Which Rules Fired?<br>"
               "<sup>Each fired rule contributes coef × 1; unfired rules contribute 0. "
               "Linear terms contribute coef × feature_value.</sup>"),
        yaxis_title="Rental count contribution",
        barmode="stack",
        xaxis_tickangle=-35,
        height=530,
        legend=dict(orientation="h", y=-0.25),
    )
    return fig


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    X_train, X_test, y_train, y_test, feature_names = load_regression()

    print("Fitting RuleFit (this may take a moment)…")
    # Use a subset + limited trees to keep runtime reasonable
    samp = np.random.default_rng(42).choice(len(X_train), size=4000, replace=False)
    model = RuleFitRegressor(n_estimators=50, tree_size=4, random_state=42)
    model.fit(X_train[samp], y_train[samp], feature_names=feature_names)
    print(f"  Model fitted. Non-zero terms: {(model._get_rules()['coef'] != 0).sum()}")

    print("Figure 1: Pipeline diagram…")
    fig1 = plot_rulefit_pipeline(model, feature_names)
    save_figure(fig1, CHAPTER, "how_rulefit_pipeline")

    print("Figure 2: Rule activation distribution…")
    fig2 = plot_rulefit_rule_activation(model, X_train, y_train, feature_names)
    save_figure(fig2, CHAPTER, "how_rulefit_rule_activation")

    print("Figure 3: Prediction decomposition waterfall…")
    fig3 = plot_rulefit_decompose(model, X_test, y_test, feature_names)
    save_figure(fig3, CHAPTER, "how_rulefit_decompose")


if __name__ == "__main__":
    main()

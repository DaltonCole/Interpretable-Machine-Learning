"""
Chapter 7: How Logistic Regression Works
https://christophm.github.io/interpretable-ml-book/logistic.html

Conceptual figures explaining the method — separate from the model-output
figures in main.py.

Figures produced:
  1. how_lr_sigmoid.html/png       — sigmoid curve fit on top feature + log-odds→prob bars
  2. how_lr_contributions.html/png — waterfall of log-odds contributions for one instance
  3. how_lr_calibration.html/png   — calibration curve + predicted probability distributions
"""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from shared.datasets import load_classification
from shared.theme import COLORS, save_figure

CHAPTER = "ch07_logistic_regression"
RNG = np.random.default_rng(42)


def fit_pipeline(X_train, y_train):
    pipe = Pipeline([("scaler", StandardScaler()), ("lr", LogisticRegression(max_iter=1000, random_state=42))])
    pipe.fit(X_train, y_train)
    return pipe


# ── Figure 1: The sigmoid turns scores into probabilities ────────────────────

def plot_lr_sigmoid(pipe, X_train, y_train, feature_names) -> go.Figure:
    """Left: raw data vs sigmoid curve on top feature.
    Right: log-odds and probability for 3 representative instances.
    """
    scaler = pipe.named_steps["scaler"]
    lr     = pipe.named_steps["lr"]
    coefs  = lr.coef_[0]
    top_idx = int(np.argmax(np.abs(coefs)))
    feat_name = feature_names[top_idx]

    # --- Left panel: sigmoid on top feature ---
    feat_vals = X_train[:, top_idx]
    feat_min, feat_max = feat_vals.min(), feat_vals.max()
    grid = np.linspace(feat_min, feat_max, 300)

    X_med = np.median(X_train, axis=0)
    X_sweep = np.tile(X_med, (300, 1))
    X_sweep[:, top_idx] = grid
    sigmoid_probs = pipe.predict_proba(X_sweep)[:, 1]

    # Jitter class labels slightly for visibility
    jitter = RNG.uniform(-0.03, 0.03, size=len(y_train))
    y_jittered = y_train.astype(float) + jitter

    # --- Right panel: log-odds → probability for 3 instances ---
    X_s = scaler.transform(X_train)
    log_odds_all = X_s @ coefs + lr.intercept_[0]
    probs_all = pipe.predict_proba(X_train)[:, 1]

    # Pick clearly benign, borderline, clearly malignant
    benign_idx     = int(np.argmin(np.abs(probs_all - 0.05)))
    borderline_idx = int(np.argmin(np.abs(probs_all - 0.50)))
    malignant_idx  = int(np.argmin(np.abs(probs_all - 0.95)))
    sel_idx = [benign_idx, borderline_idx, malignant_idx]
    sel_labels = ["Clearly benign", "Borderline", "Clearly malignant"]
    sel_logodds = [log_odds_all[i] for i in sel_idx]
    sel_probs   = [probs_all[i]    for i in sel_idx]
    instance_colors = [COLORS["positive"], COLORS["neutral"], COLORS["negative"]]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[
            f"Sigmoid Fit on '{feat_name}'",
            "Log-odds → Probability",
        ],
        column_widths=[0.55, 0.45],
        horizontal_spacing=0.12,
    )

    # Raw scatter — actual class
    samp = RNG.choice(len(X_train), size=min(500, len(X_train)), replace=False)
    for cls, cls_label, col in [(0, "Benign (actual)", COLORS["positive"]),
                                 (1, "Malignant (actual)", COLORS["negative"])]:
        mask = y_train[samp] == cls
        fig.add_trace(go.Scatter(
            x=X_train[samp][mask, top_idx],
            y=y_jittered[samp][mask],
            mode="markers",
            marker=dict(color=col, size=5, opacity=0.35),
            name=cls_label,
        ), row=1, col=1)

    # Sigmoid curve
    fig.add_trace(go.Scatter(
        x=grid, y=sigmoid_probs,
        mode="lines",
        line=dict(color=COLORS["primary"], width=3),
        name="P(malignant) sigmoid",
    ), row=1, col=1)

    fig.add_hline(y=0.5, line_dash="dash", line_color=COLORS["neutral"],
                  line_width=1.5, row=1, col=1)
    fig.add_annotation(
        row=1, col=1,
        x=feat_max * 0.75, y=0.53,
        text="Decision boundary (p = 0.5)",
        showarrow=False, font=dict(size=11, color=COLORS["neutral"]),
        bgcolor="rgba(255,255,255,0.9)", bordercolor=COLORS["neutral"], borderwidth=1,
    )

    fig.update_xaxes(title_text=feat_name, row=1, col=1)
    fig.update_yaxes(title_text="P(malignant) / class label", row=1, col=1)

    # Right panel: grouped bars for log-odds and probability
    # Use two sub-groups: show log-odds on left, prob on right, per instance
    bar_x_lo = [f"{lbl}<br>(log-odds)" for lbl in sel_labels]
    bar_x_pr = [f"{lbl}<br>(prob)" for lbl in sel_labels]

    fig.add_trace(go.Bar(
        x=sel_labels,
        y=sel_logodds,
        name="Log-odds (linear score)",
        marker_color=instance_colors,
        opacity=0.75,
        showlegend=False,
        hovertemplate="%{x}<br>Log-odds: %{y:.2f}<extra></extra>",
        text=[f"{v:.2f}" for v in sel_logodds],
        textposition="outside",
    ), row=1, col=2)

    # Overlay probability as a secondary group by using offset scatter-markers
    for i, (lbl, lo, pr, col) in enumerate(zip(sel_labels, sel_logodds, sel_probs, instance_colors)):
        fig.add_annotation(
            row=1, col=2,
            x=lbl, y=min(sel_logodds) - 1.8 - i * 0.05,
            text=f"→ P = {pr:.2f}",
            showarrow=False,
            font=dict(size=12, color=col),
        )

    fig.add_hline(y=0, line_dash="dot", line_color=COLORS["neutral"],
                  line_width=1, row=1, col=2)
    fig.update_xaxes(title_text="Instance type", row=1, col=2)
    fig.update_yaxes(title_text="Log-odds", row=1, col=2)

    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.98, y=0.98,
        text="σ(z) = 1 / (1 + e<sup>−z</sup>)",
        showarrow=False, align="right",
        font=dict(size=13, color=COLORS["primary"]),
        bgcolor="rgba(255,255,255,0.9)", bordercolor=COLORS["primary"], borderwidth=1,
    )

    fig.update_layout(
        title=("Logistic Regression — The Sigmoid Turns Scores into Probabilities<br>"
               "<sup>A linear score (log-odds) is passed through σ(z) = 1/(1+e<sup>−z</sup>) "
               "to produce a probability bounded in [0, 1].</sup>"),
        height=500,
        legend=dict(orientation="h", y=-0.18),
    )
    return fig


# ── Figure 2: Log-odds are a weighted sum of features ────────────────────────

def plot_lr_contributions(pipe, X_train, X_test, y_test, feature_names) -> go.Figure:
    """Waterfall decomposing one prediction's log-odds into per-feature contributions.

    contribution_i = coef_i * (x_i - mean_i) / std_i   (log-odds units)
    """
    scaler = pipe.named_steps["scaler"]
    lr     = pipe.named_steps["lr"]
    coefs  = lr.coef_[0]

    # Pick the test instance predicted MOST CONFIDENTLY as malignant
    probs = pipe.predict_proba(X_test)[:, 1]
    inst_idx  = int(np.argmax(probs))
    instance  = X_test[inst_idx]
    prob_pred = probs[inst_idx]

    # Contributions in log-odds space: coef_i * z_i where z_i is standardised value
    z = scaler.transform(instance.reshape(1, -1))[0]
    contributions = coefs * z
    intercept = float(lr.intercept_[0])
    log_odds_final = intercept + contributions.sum()

    # Top 10 by absolute contribution
    order = np.argsort(np.abs(contributions))[::-1][:10]
    contrib_top = contributions[order]
    names_top   = [feature_names[i] for i in order]
    vals_top    = [instance[i] for i in order]

    # Build waterfall
    running = intercept
    x_labels  = ["Intercept"]
    bar_vals   = [intercept]
    bar_bases  = [0]
    bar_colors = [COLORS["neutral"]]
    hover_texts = [f"Mean log-odds (intercept) = {intercept:.3f}"]

    for name, contrib, feat_val in zip(names_top, contrib_top, vals_top):
        x_labels.append(name)
        bar_vals.append(contrib)
        bar_bases.append(running)
        bar_colors.append(COLORS["positive"] if contrib >= 0 else COLORS["negative"])
        hover_texts.append(f"<b>{name}</b> = {feat_val:.3f}<br>Contribution: {contrib:+.3f} log-odds")
        running += contrib

    x_labels.append("Log-odds")
    bar_vals.append(log_odds_final)
    bar_bases.append(0)
    bar_colors.append(COLORS["primary"])
    hover_texts.append(f"Final log-odds = {log_odds_final:.3f}<br>P(malignant) = {prob_pred:.3f}")

    fig = go.Figure()

    for lbl, val, base, col, htxt in zip(x_labels[:-1], bar_vals[:-1], bar_bases[:-1],
                                          bar_colors[:-1], hover_texts[:-1]):
        fig.add_trace(go.Bar(
            x=[lbl], y=[val], base=[base],
            marker_color=col,
            showlegend=False,
            hovertemplate=htxt + "<extra></extra>",
            texttemplate=f"{val:+.3f}",
            textposition="outside",
        ))

    fig.add_trace(go.Bar(
        x=["Log-odds"], y=[log_odds_final], base=[0],
        marker_color=COLORS["primary"],
        showlegend=False,
        hovertemplate=hover_texts[-1] + "<extra></extra>",
        texttemplate=f"{log_odds_final:.3f}",
        textposition="outside",
    ))

    fig.add_hline(y=intercept, line_dash="dash", line_color=COLORS["neutral"], line_width=1)
    fig.add_hline(y=0, line_dash="dot", line_color=COLORS["neutral"], line_width=1)

    fig.add_annotation(
        x="Log-odds", y=log_odds_final + 0.3,
        text=f"P(malignant) = σ({log_odds_final:.2f}) = <b>{prob_pred:.2f}</b>",
        showarrow=True, arrowhead=2, ax=80, ay=-40,
        font=dict(size=13, color=COLORS["primary"]),
        bgcolor="rgba(255,255,255,0.9)", bordercolor=COLORS["primary"], borderwidth=1,
    )

    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.01, y=0.01,
        text="exp(log-odds) = odds &nbsp;|&nbsp; P = odds / (1 + odds)",
        showarrow=False, align="left",
        font=dict(size=11, color=COLORS["neutral"]),
        bgcolor="rgba(255,255,255,0.85)", bordercolor="#D1D5DB", borderwidth=1,
    )

    # Legend entries
    fig.add_trace(go.Bar(x=[None], y=[None], marker_color=COLORS["positive"], name="Increases log-odds (→ malignant)"))
    fig.add_trace(go.Bar(x=[None], y=[None], marker_color=COLORS["negative"], name="Decreases log-odds (→ benign)"))
    fig.add_trace(go.Bar(x=[None], y=[None], marker_color=COLORS["neutral"],  name="Intercept (baseline)"))

    fig.update_layout(
        title=("Logistic Regression — Log-Odds Are a Weighted Sum of Features<br>"
               f"<sup>Instance #{inst_idx}: log-odds = {log_odds_final:.2f} → "
               f"P(malignant) = {prob_pred:.2f} &nbsp;|&nbsp; "
               f"Actual: {'malignant' if y_test[inst_idx] == 1 else 'benign'}</sup>"),
        yaxis_title="Log-odds contribution",
        barmode="stack",
        xaxis_tickangle=-30,
        height=520,
        legend=dict(orientation="h", y=-0.22),
    )
    return fig


# ── Figure 3: Is the model well-calibrated? ──────────────────────────────────

def plot_lr_calibration(pipe, X_test, y_test) -> go.Figure:
    """Left: calibration curve. Right: predicted probability distributions by class."""
    probs = pipe.predict_proba(X_test)[:, 1]

    # Calibration: bin into 10 buckets
    n_bins = 10
    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers, mean_pred, frac_pos = [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() > 0:
            bin_centers.append((lo + hi) / 2)
            mean_pred.append(float(probs[mask].mean()))
            frac_pos.append(float(y_test[mask].mean()))

    mean_pred = np.array(mean_pred)
    frac_pos  = np.array(frac_pos)
    bin_centers = np.array(bin_centers)

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Calibration Curve", "Predicted Probability by Class"],
        column_widths=[0.5, 0.5],
        horizontal_spacing=0.12,
    )

    # --- Left: calibration ---
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines",
        line=dict(dash="dash", color=COLORS["neutral"], width=2),
        name="Perfect calibration",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=mean_pred, y=frac_pos,
        mode="lines+markers",
        line=dict(color=COLORS["primary"], width=2.5),
        marker=dict(size=9, color=COLORS["primary"]),
        name="Logistic regression",
        hovertemplate="Mean predicted: %{x:.2f}<br>Actual fraction: %{y:.2f}<extra></extra>",
    ), row=1, col=1)

    fig.update_xaxes(title_text="Mean predicted probability", range=[-0.05, 1.05], row=1, col=1)
    fig.update_yaxes(title_text="Fraction of positives (malignant)", range=[-0.05, 1.05], row=1, col=1)

    fig.add_annotation(
        row=1, col=1,
        xref="x domain", yref="y domain",
        x=0.05, y=0.95,
        text="Points on diagonal → perfect calibration<br>Above = overestimates risk",
        showarrow=False, align="left", xanchor="left", yanchor="top",
        font=dict(size=11),
        bgcolor="rgba(255,255,255,0.85)", bordercolor="#D1D5DB", borderwidth=1,
    )

    # --- Right: probability distributions by class ---
    for cls, cls_name, col in [(0, "Benign", COLORS["positive"]),
                                (1, "Malignant", COLORS["negative"])]:
        mask = y_test == cls
        fig.add_trace(go.Histogram(
            x=probs[mask],
            nbinsx=20,
            name=cls_name,
            marker_color=col,
            opacity=0.65,
            histnorm="probability density",
            hovertemplate=f"{cls_name}<br>P(malignant)=%{{x:.2f}}<extra></extra>",
        ), row=1, col=2)

    fig.add_vline(x=0.5, line_dash="dash", line_color=COLORS["neutral"],
                  line_width=1.5, row=1, col=2)
    fig.add_annotation(
        row=1, col=2,
        xref="x2 domain", yref="y2 domain",
        x=0.52, y=0.97,
        text="Decision threshold",
        showarrow=False, align="left", xanchor="left", yanchor="top",
        font=dict(size=11, color=COLORS["neutral"]),
    )

    fig.update_xaxes(title_text="Predicted P(malignant)", row=1, col=2)
    fig.update_yaxes(title_text="Density", row=1, col=2)

    fig.update_layout(
        title=("Logistic Regression — Is the Model Well-Calibrated?<br>"
               "<sup>Calibration: does P=0.7 mean 70% of those instances are actually positive? "
               "Good separation = classes cluster at opposite ends.</sup>"),
        barmode="overlay",
        height=500,
        legend=dict(orientation="h", y=-0.18),
    )
    return fig


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    X_train, X_test, y_train, y_test, feature_names, class_names = load_classification()
    pipe = fit_pipeline(X_train, y_train)

    print("Figure 1: Sigmoid fit and log-odds→probability…")
    fig1 = plot_lr_sigmoid(pipe, X_train, y_train, feature_names)
    save_figure(fig1, CHAPTER, "how_lr_sigmoid")

    print("Figure 2: Log-odds contribution waterfall…")
    fig2 = plot_lr_contributions(pipe, X_train, X_test, y_test, feature_names)
    save_figure(fig2, CHAPTER, "how_lr_contributions")

    print("Figure 3: Calibration…")
    fig3 = plot_lr_calibration(pipe, X_test, y_test)
    save_figure(fig3, CHAPTER, "how_lr_calibration")


if __name__ == "__main__":
    main()

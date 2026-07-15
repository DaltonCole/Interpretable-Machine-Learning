"""
Chapter 14: How LIME Works
https://christophm.github.io/interpretable-ml-book/lime.html

Conceptual figures explaining Local Interpretable Model-Agnostic Explanations.
Uses a manual Ridge-regression approximation instead of the lime library so
there are no extra dependencies and every step is visible in the code.

Figures produced:
  1. how_lime_neighborhood.html/png — PCA 2D view of local sampling + distance weights
  2. how_lime_local_model.html/png  — feature scatter + local Ridge explanation bar chart
  3. how_lime_stability.html/png    — box plots showing explanation variance across 10 runs
"""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from shared.datasets import load_regression
from shared.models import get_regression_model
from shared.theme import COLORS, save_figure

CHAPTER = "ch14_lime"
RNG = np.random.default_rng(42)

N_PERTURB = 200      # neighbourhood samples per run
N_STABILITY = 10     # repeated runs for stability plot
TOP_N_FEATURES = 5   # features to show in explanation bar chart


# ── Neighbourhood sampling helpers ────────────────────────────────────────────

def sample_neighbourhood(instance, X_train, n_samples, rng):
    """Normal perturbations around `instance` clipped to training range."""
    sigma = X_train.std(axis=0) * 0.35
    X_perturb = rng.normal(loc=instance, scale=sigma, size=(n_samples, instance.shape[0]))
    for j in range(instance.shape[0]):
        X_perturb[:, j] = np.clip(X_perturb[:, j],
                                   X_train[:, j].min(),
                                   X_train[:, j].max())
    return X_perturb


def proximity_weights(X_perturb, instance, X_train):
    """Exponential kernel: weight = exp(-d² / 2σ²), d = normalised Euclidean."""
    std = X_train.std(axis=0) + 1e-8
    kernel_width = 0.75 * np.sqrt(instance.shape[0])
    dists = np.linalg.norm((X_perturb - instance) / std, axis=1)
    return np.exp(-dists ** 2 / (2 * kernel_width ** 2))


def fit_local_ridge(X_perturb, y_perturb, weights):
    """Fit a weighted Ridge regression; return (model, scaler)."""
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X_perturb)
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_std, y_perturb, sample_weight=weights)
    return ridge, scaler


# ── Figure 1: PCA neighbourhood scatter ──────────────────────────────────────

def plot_lime_neighborhood(model, instance, X_train, feature_names) -> go.Figure:
    """PCA 2D projection showing training data, perturbed samples (coloured by
    proximity weight), and the target instance as a star."""
    X_perturb = sample_neighbourhood(instance, X_train, N_PERTURB, RNG)
    y_perturb = model.predict(X_perturb)
    weights = proximity_weights(X_perturb, instance, X_train)

    # PCA fit on training data
    pca = PCA(n_components=2, random_state=42)
    pca.fit(X_train)

    # Background training data (500 random points)
    bg_idx = RNG.choice(len(X_train), size=500, replace=False)
    X_bg_2d = pca.transform(X_train[bg_idx])
    X_perturb_2d = pca.transform(X_perturb)
    instance_2d = pca.transform(instance.reshape(1, -1))[0]

    fig = go.Figure()

    # Training data background
    fig.add_trace(go.Scatter(
        x=X_bg_2d[:, 0], y=X_bg_2d[:, 1],
        mode="markers",
        marker=dict(color=COLORS["neutral"], size=4, opacity=0.20),
        name="Training data",
        hoverinfo="skip",
    ))

    # Perturbed samples coloured by proximity weight
    fig.add_trace(go.Scatter(
        x=X_perturb_2d[:, 0], y=X_perturb_2d[:, 1],
        mode="markers",
        marker=dict(
            color=weights,
            colorscale="Blues",
            showscale=False,     # avoid colorbar overlap
            size=7,
            opacity=0.80,
            line=dict(color="white", width=0.3),
        ),
        name="Perturbed samples",
        hovertemplate="Weight=%{marker.color:.3f}<extra></extra>",
    ))

    # Target instance
    fig.add_trace(go.Scatter(
        x=[instance_2d[0]], y=[instance_2d[1]],
        mode="markers",
        marker=dict(color=COLORS["accent"], size=18, symbol="star",
                    line=dict(color="white", width=2)),
        name="Target instance",
    ))

    # Inline legend note (no colorbar)
    fig.add_annotation(
        x=0.02, y=0.97,
        xref="paper", yref="paper",
        text="● darker blue = closer to instance (higher weight)",
        showarrow=False, align="left", xanchor="left", yanchor="top",
        font=dict(size=12),
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor=COLORS["primary"], borderwidth=1,
    )

    ev = pca.explained_variance_ratio_
    fig.update_layout(
        title=("LIME — Sample a Local Neighbourhood and Weight by Proximity<br>"
               "<sup>Perturbed samples drawn around the target instance (star). "
               "Closer samples get higher weights in the local fit.</sup>"),
        xaxis_title=f"PC1 ({ev[0]:.1%} variance)",
        yaxis_title=f"PC2 ({ev[1]:.1%} variance)",
        height=500,
        legend=dict(orientation="h", y=-0.18),
    )
    return fig


# ── Figure 2: local linear model ─────────────────────────────────────────────

def plot_lime_local_model(model, instance, X_train, feature_names) -> go.Figure:
    """Left: scatter of most-important LIME feature vs model predictions (weighted).
    Right: horizontal bar chart of local Ridge coefficients."""
    X_perturb = sample_neighbourhood(instance, X_train, N_PERTURB, RNG)
    y_perturb = model.predict(X_perturb)
    weights = proximity_weights(X_perturb, instance, X_train)
    ridge, scaler = fit_local_ridge(X_perturb, y_perturb, weights)

    coefs = ridge.coef_  # standardised coefficients
    top_idx = np.argsort(np.abs(coefs))[-TOP_N_FEATURES:][::-1]
    best_feat_idx = top_idx[0]  # most important single feature for scatter

    # Prediction line: vary best feature, hold others at instance value
    x_range = np.linspace(X_perturb[:, best_feat_idx].min(),
                          X_perturb[:, best_feat_idx].max(), 100)
    X_line = np.tile(instance, (100, 1))
    X_line[:, best_feat_idx] = x_range
    y_line_ridge = ridge.predict(scaler.transform(X_line))

    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.50, 0.50],
        subplot_titles=[
            f"Perturbed samples: '{feature_names[best_feat_idx]}' vs prediction",
            "Local Linear Explanation (top features)",
        ],
        horizontal_spacing=0.14,
    )

    # ── left: scatter + local linear fit ──
    # Clip marker sizes to a reasonable range
    marker_sizes = 4 + weights * 14

    fig.add_trace(go.Scatter(
        x=X_perturb[:, best_feat_idx], y=y_perturb,
        mode="markers",
        marker=dict(
            color=weights,
            colorscale="Blues",
            showscale=False,
            size=marker_sizes,
            opacity=0.75,
            line=dict(color="white", width=0.3),
        ),
        name="Perturbed samples (size ∝ weight)",
        hovertemplate=f"{feature_names[best_feat_idx]}=%{{x:.2f}}<br>Pred=%{{y:.0f}}<extra></extra>",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=x_range, y=y_line_ridge,
        mode="lines",
        line=dict(color=COLORS["accent"], width=2.5),
        name="Local linear approximation",
    ), row=1, col=1)

    # Mark actual instance
    actual_val = float(instance[best_feat_idx])
    actual_pred = float(model.predict(instance.reshape(1, -1))[0])
    fig.add_trace(go.Scatter(
        x=[actual_val], y=[actual_pred],
        mode="markers",
        marker=dict(size=14, color=COLORS["accent"], symbol="star",
                    line=dict(color="white", width=2)),
        name="Target instance",
    ), row=1, col=1)

    fig.add_annotation(
        x=0.01, y=0.99, xref="paper", yref="paper",
        text="Global black-box model (complex)<br>→ Local linear approximation (interpretable)",
        showarrow=False, align="left", xanchor="left", yanchor="top",
        font=dict(size=11),
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor=COLORS["neutral"], borderwidth=1,
    )

    # ── right: coefficient bar chart ──
    top_names = [feature_names[i] for i in top_idx]
    top_coefs = coefs[top_idx]
    bar_colors = [COLORS["positive"] if c > 0 else COLORS["negative"] for c in top_coefs]

    fig.add_trace(go.Bar(
        x=top_coefs,
        y=top_names,
        orientation="h",
        marker_color=bar_colors,
        name="LIME feature weight",
        hovertemplate="%{y}: %{x:+.3f}<extra></extra>",
    ), row=1, col=2)

    fig.add_vline(x=0, line_color=COLORS["neutral"], line_width=1)

    fig.update_xaxes(title_text=feature_names[best_feat_idx], row=1, col=1)
    fig.update_yaxes(title_text="Predicted rentals", row=1, col=1)
    fig.update_xaxes(title_text="Standardised coefficient", row=1, col=2)

    fig.update_layout(
        title=("LIME — Fit a Simple Linear Model in the Local Neighbourhood<br>"
               "<sup>Larger/darker dots = higher weight (closer to instance). "
               "The pink line is the local linear approximation LIME uses.</sup>"),
        height=490,
        legend=dict(orientation="h", y=-0.18),
    )
    return fig


# ── Figure 3: stability across repeated runs ──────────────────────────────────

def plot_lime_stability(model, instance, X_train, feature_names) -> go.Figure:
    """Box plot of top-5 feature weights across 10 random-seed runs.

    Wide spread = unstable explanation for that feature.
    """
    # Reference run to identify the top-5 features
    X_ref = sample_neighbourhood(instance, X_train, N_PERTURB, RNG)
    y_ref = model.predict(X_ref)
    w_ref = proximity_weights(X_ref, instance, X_train)
    ridge_ref, _ = fit_local_ridge(X_ref, y_ref, w_ref)
    mean_abs = np.abs(ridge_ref.coef_)
    top5_idx = np.argsort(mean_abs)[-5:][::-1]

    # Repeated runs
    all_coefs = []  # shape (N_STABILITY, n_features)
    for seed in range(N_STABILITY):
        rng_s = np.random.default_rng(seed + 100)
        X_s = sample_neighbourhood(instance, X_train, N_PERTURB, rng_s)
        y_s = model.predict(X_s)
        w_s = proximity_weights(X_s, instance, X_train)
        ridge_s, _ = fit_local_ridge(X_s, y_s, w_s)
        all_coefs.append(ridge_s.coef_)

    all_coefs = np.array(all_coefs)  # (10, n_features)

    # Compute coefficient of variation (cv) to flag unstable features
    coef_std = all_coefs[:, top5_idx].std(axis=0)
    coef_mean_abs = np.abs(all_coefs[:, top5_idx]).mean(axis=0) + 1e-8
    cv = coef_std / coef_mean_abs  # higher = more unstable

    top5_names = [feature_names[i] for i in top5_idx]

    fig = go.Figure()
    for rank, (feat_idx, feat_name) in enumerate(zip(top5_idx, top5_names)):
        is_unstable = cv[rank] > 0.3  # flag high-CV features
        box_color = COLORS["negative"] if is_unstable else COLORS["primary"]
        fig.add_trace(go.Box(
            y=all_coefs[:, feat_idx],
            name=feat_name,
            boxpoints="all",
            jitter=0.35,
            marker=dict(size=5, color=box_color, opacity=0.7),
            line=dict(color=box_color),
            fillcolor=f"rgba({','.join(str(int(c, 16)) for c in [box_color[1:3], box_color[3:5], box_color[5:7]])},0.12)",
        ))

    fig.add_hline(y=0, line_dash="dot", line_color=COLORS["neutral"], line_width=1)

    # Annotation: wide spread = uncertain
    widest = top5_names[int(np.argmax(cv))]
    fig.add_annotation(
        x=0.98, y=0.97,
        xref="paper", yref="paper",
        text=f"Wide spread = this feature's importance is uncertain<br>"
             f"(e.g. <b>{widest}</b> has high variability across runs)",
        showarrow=False,
        align="right", xanchor="right", yanchor="top",
        font=dict(size=12),
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor=COLORS["negative"], borderwidth=1,
    )

    fig.update_layout(
        title=("LIME — Stability: Do Different Samples Give the Same Explanation?<br>"
               "<sup>Each box = distribution of a feature's weight across 10 random-sample runs. "
               "Wide box = unstable, narrow = stable explanation.</sup>"),
        yaxis_title="Standardised coefficient (local Ridge)",
        xaxis_title="Feature",
        showlegend=False,
        height=490,
    )
    return fig


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    X_train, X_test, y_train, y_test, feature_names = load_regression()
    model = get_regression_model(X_train, y_train)

    instance = X_test[0]
    pred = float(model.predict(instance.reshape(1, -1))[0])
    print(f"Instance prediction: {pred:.0f}")

    print("Figure 1: LIME neighbourhood (PCA)…")
    fig1 = plot_lime_neighborhood(model, instance, X_train, feature_names)
    save_figure(fig1, CHAPTER, "how_lime_neighborhood")

    print("Figure 2: LIME local model…")
    fig2 = plot_lime_local_model(model, instance, X_train, feature_names)
    save_figure(fig2, CHAPTER, "how_lime_local_model")

    print("Figure 3: LIME stability…")
    fig3 = plot_lime_stability(model, instance, X_train, feature_names)
    save_figure(fig3, CHAPTER, "how_lime_stability")


if __name__ == "__main__":
    main()

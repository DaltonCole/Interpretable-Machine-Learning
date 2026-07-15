"""
Chapter 26: How Prototypes and Criticisms Work
https://christophm.github.io/interpretable-ml-book/proto.html

Figures produced:
  1. how_proto_concept.html/png      — k-medoids prototypes in PCA space
  2. how_criticism_concept.html/png  — criticisms as outliers far from prototypes
  3. how_proto_explanation.html/png  — parallel coordinates comparing instance to its prototype
"""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from shared.datasets import load_regression
from shared.models import get_regression_model
from shared.theme import COLORS, save_figure

CHAPTER = "ch26_prototypes"
RNG = np.random.default_rng(42)

N_PROTOTYPES = 5


def _kmedoids(X, k, rng, n_iter=50):
    """Simple k-medoids: random init, alternating assignment and medoid update."""
    n = len(X)
    medoid_idx = rng.choice(n, size=k, replace=False)
    for _ in range(n_iter):
        dists = np.array([np.linalg.norm(X - X[m], axis=1) for m in medoid_idx])
        labels = np.argmin(dists, axis=0)
        new_medoids = []
        for c in range(k):
            members = np.where(labels == c)[0]
            if len(members) == 0:
                new_medoids.append(medoid_idx[c])
                continue
            within = np.sum(np.linalg.norm(X[members][:, None] - X[members][None, :], axis=2), axis=1)
            new_medoids.append(members[np.argmin(within)])
        medoid_idx = np.array(new_medoids)
    # final assignment
    dists = np.array([np.linalg.norm(X - X[m], axis=1) for m in medoid_idx])
    labels = np.argmin(dists, axis=0)
    return medoid_idx, labels


# ── Figure 1: k-medoids prototypes in PCA space ────────────────────────────

def plot_proto_concept(X_train, y_train, feature_names, medoid_idx, all_labels, X_2d, X_std):
    palette = COLORS["palette"]

    # Sample for display
    samp = RNG.choice(len(X_2d), size=1500, replace=False)

    fig = go.Figure()

    # Scatter — cluster membership for sampled points

    for c in range(N_PROTOTYPES):
        mask = all_labels[samp] == c
        fig.add_trace(go.Scatter(
            x=X_2d[samp[mask], 0], y=X_2d[samp[mask], 1],
            mode="markers",
            marker=dict(color=palette[c % len(palette)], size=4, opacity=0.35),
            name=f"Cluster {c+1}",
            hovertemplate="PC1=%{x:.2f}, PC2=%{y:.2f}<extra></extra>",
        ))

    # Prototypes as gold stars
    hr_idx   = feature_names.index("hr")
    temp_idx = feature_names.index("temp")
    season_idx = feature_names.index("season")

    for i, mi in enumerate(medoid_idx):
        hr_val     = X_train[mi, hr_idx]
        temp_val   = X_train[mi, temp_idx]
        season_val = X_train[mi, season_idx]
        fig.add_trace(go.Scatter(
            x=[X_2d[mi, 0]], y=[X_2d[mi, 1]],
            mode="markers+text",
            marker=dict(symbol="star", color="gold", size=18,
                        line=dict(color="black", width=1)),
            text=[f"P{i+1}"],
            textposition="top center",
            textfont=dict(size=12, color="black"),
            name=f"Prototype {i+1}",
            showlegend=False,
            hovertemplate=(
                f"<b>Prototype {i+1}</b><br>"
                f"hr={hr_val:.0f}, temp={temp_val:.2f}, season={season_val:.0f}"
                "<extra></extra>"
            ),
        ))
        fig.add_annotation(
            x=X_2d[mi, 0], y=X_2d[mi, 1] - 0.5,
            text=f"hr={hr_val:.0f} temp={temp_val:.2f}",
            showarrow=False, font=dict(size=9, color="black"),
            bgcolor="rgba(255,255,240,0.9)", bordercolor="goldenrod", borderwidth=1,
        )

    fig.update_layout(
        title=("Prototypes — Representative Examples That Summarise the Data<br>"
               "<sup>k-medoids selects {k} real instances (★) that best represent each cluster. "
               "Each prototype is the most central member of its group.</sup>".format(k=N_PROTOTYPES)),
        xaxis_title="PC 1", yaxis_title="PC 2",
        height=520, legend=dict(orientation="h", y=-0.18),
    )
    return fig


# ── Figure 2: Criticisms as outliers far from prototypes ──────────────────

def plot_criticism_concept(X_train, y_train, feature_names, medoid_idx, all_labels, X_2d, X_std):
    dists_all = np.array([np.linalg.norm(X_std - X_std[m], axis=1) for m in medoid_idx])
    min_dist_to_proto = dists_all.min(axis=0)
    top_k_crit = np.argsort(min_dist_to_proto)[-10:][::-1]

    palette = COLORS["palette"]
    samp = RNG.choice(len(X_2d), size=1500, replace=False)

    fig = go.Figure()

    for c in range(N_PROTOTYPES):
        mask = all_labels[samp] == c
        fig.add_trace(go.Scatter(
            x=X_2d[samp[mask], 0], y=X_2d[samp[mask], 1],
            mode="markers",
            marker=dict(color=palette[c % len(palette)], size=4, opacity=0.25),
            name=f"Cluster {c+1}",
            hovertemplate="PC1=%{x:.2f}<br>PC2=%{y:.2f}<extra></extra>",
        ))

    # Prototypes
    fig.add_trace(go.Scatter(
        x=X_2d[medoid_idx, 0], y=X_2d[medoid_idx, 1],
        mode="markers+text",
        marker=dict(symbol="star", color="gold", size=18, line=dict(color="black", width=1)),
        text=[f"P{i+1}" for i in range(N_PROTOTYPES)],
        textposition="top center",
        name="Prototypes (★)",
        hovertemplate="Prototype<extra></extra>",
    ))

    # Draw lines from criticisms to their nearest prototype
    for ci in top_k_crit:
        nearest_p_idx = int(np.argmin([np.linalg.norm(X_std[ci] - X_std[m]) for m in medoid_idx]))
        nearest_proto = medoid_idx[nearest_p_idx]
        fig.add_shape(type="line",
            x0=X_2d[ci, 0], y0=X_2d[ci, 1],
            x1=X_2d[nearest_proto, 0], y1=X_2d[nearest_proto, 1],
            line=dict(color=COLORS["negative"], width=1, dash="dot"))

    # Criticisms
    fig.add_trace(go.Scatter(
        x=X_2d[top_k_crit, 0], y=X_2d[top_k_crit, 1],
        mode="markers",
        marker=dict(symbol="x", color=COLORS["negative"], size=14, line=dict(width=2)),
        name="Criticisms (✕)",
        hovertemplate="Criticism — far from all prototypes<extra></extra>",
    ))

    fig.add_annotation(
        x=0.98, y=0.96, xref="paper", yref="paper",
        xanchor="right", yanchor="top",
        text="Criticisms are the instances<br>that no prototype captures well.",
        showarrow=False, font=dict(size=12),
        bgcolor="rgba(255,255,255,0.9)", bordercolor=COLORS["negative"], borderwidth=1,
    )

    fig.update_layout(
        title=("Criticisms — Instances the Prototypes Fail to Represent<br>"
               "<sup>Criticisms (✕) are the training instances farthest from any prototype (★). "
               "They are outliers that need their own explanation.</sup>"),
        xaxis_title="PC 1", yaxis_title="PC 2",
        height=520, legend=dict(orientation="h", y=-0.18),
    )
    return fig


# ── Figure 3: Instance vs prototype in parallel coordinates ──────────────

def plot_proto_explanation(X_train, y_train, feature_names, medoid_idx, X_std):
    dists_all = np.array([np.linalg.norm(X_std - X_std[m], axis=1) for m in medoid_idx])
    all_labels_full = np.argmin(dists_all, axis=0)

    # Pick 3 diverse test instances: low pred, mid pred, high pred
    preds = y_train
    low_i  = int(np.argmin(preds))
    high_i = int(np.argmax(preds))
    mid_i  = int(np.argmin(np.abs(preds - np.median(preds))))
    instances = [low_i, mid_i, high_i]
    labels_inst = ["Low rentals", "Median rentals", "High rentals"]

    # Top 6 features by variance (most informative for comparison)
    top_feats = np.argsort(X_train.std(axis=0))[::-1][:6]
    feat_names_top = [feature_names[i] for i in top_feats]

    # Build parallel coordinates lines
    dimensions = []
    for j, fi in enumerate(top_feats):
        vals = X_train[:, fi]
        vmin, vmax = vals.min(), vals.max()
        dim_vals = []
        for inst_i in instances:
            dim_vals.append(float(X_train[inst_i, fi]))
            proto_i = medoid_idx[all_labels_full[inst_i]]
            dim_vals.append(float(X_train[proto_i, fi]))
        dimensions.append(dict(
            label=feature_names[fi],
            values=dim_vals,
            range=[vmin, vmax],
        ))

    # Color: alternate instance/prototype pairs
    colors = []
    color_map = {0: 0, 1: 0.5, 2: 1.0}
    pair_color = [0, 0, 0.5, 0.5, 1.0, 1.0]

    fig = go.Figure(go.Parcoords(
        line=dict(
            color=pair_color,
            colorscale=[[0, COLORS["primary"]], [0.5, COLORS["accent"]], [1, COLORS["positive"]]],
            showscale=False,
        ),
        dimensions=dimensions,
        labelangle=-20,
        labelside="bottom",
    ))

    # Add annotation describing the line pairs
    fig.add_annotation(
        x=0.5, y=1.12, xref="paper", yref="paper",
        text=(
            "<span style='color:#2563EB'>──</span> Low-rental instance &nbsp; "
            "<span style='color:#2563EB'>- -</span> its prototype &nbsp;|&nbsp; "
            "<span style='color:#DB2777'>──</span> Median instance &nbsp;|&nbsp; "
            "<span style='color:#059669'>──</span> High-rental instance"
        ),
        showarrow=False, align="center", font=dict(size=11),
        bgcolor="rgba(255,255,255,0.9)",
    )

    fig.update_layout(
        title=("Prototypes as Explanations — 'Your Case Is Similar to This Typical Case'<br>"
               "<sup>For 3 test instances (low / median / high rentals), we show their feature values "
               "alongside their nearest prototype. Overlap = similarity; divergence = what makes the instance unusual.</sup>"),
        height=520, margin=dict(t=120, b=80),
    )
    return fig


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    X_train, X_test, y_train, y_test, feature_names = load_regression()

    scaler = StandardScaler()
    X_std  = scaler.fit_transform(X_train)
    pca    = PCA(n_components=2, random_state=42)
    X_2d   = pca.fit_transform(X_std)

    print("Computing k-medoids prototypes…")
    medoid_idx, _ = _kmedoids(X_std, N_PROTOTYPES, RNG, n_iter=30)
    dists_all  = np.array([np.linalg.norm(X_std - X_std[m], axis=1) for m in medoid_idx])
    all_labels = np.argmin(dists_all, axis=0)

    print("Figure 1: Prototypes in PCA space…")
    fig1 = plot_proto_concept(X_train, y_train, feature_names, medoid_idx, all_labels, X_2d, X_std)
    save_figure(fig1, CHAPTER, "how_proto_concept")

    print("Figure 2: Criticisms…")
    fig2 = plot_criticism_concept(X_train, y_train, feature_names, medoid_idx, all_labels, X_2d, X_std)
    save_figure(fig2, CHAPTER, "how_criticism_concept")

    print("Figure 3: Prototype-based explanation (parallel coords)…")
    fig3 = plot_proto_explanation(X_train, y_train, feature_names, medoid_idx, X_std)
    save_figure(fig3, CHAPTER, "how_proto_explanation")


if __name__ == "__main__":
    main()

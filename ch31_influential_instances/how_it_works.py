"""
Chapter 31: How Influential Instances Work
https://christophm.github.io/interpretable-ml-book/influential.html

Conceptual figures explaining influence functions, Cook's distance, and TracIn.

Figures produced:
  1. how_influence_concept.html/png — regression line shifts when 3 high-influence points
                                      are removed (bike sharing / temp feature)
  2. how_cooks_concept.html/png     — leverage vs |residual| scatter, sized by Cook's D
  3. how_tracin_concept.html/png    — test image + top-5 most helpful / most harmful
                                      training images by TracIn score
"""
import numpy as np
import plotly.graph_objects as go
import torch
import torch.nn.functional as F
from plotly.subplots import make_subplots
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Subset

from ch27_learned_features.main import SimpleCNN, train_model
from shared.datasets import load_images, load_regression
from shared.theme import COLORS, save_figure

CHAPTER = "ch31_influential_instances"
RNG = np.random.default_rng(42)


# ── Shared helpers ────────────────────────────────────────────────────────────

def _hat_diagonal(X: np.ndarray) -> np.ndarray:
    """Diagonal of the hat matrix H = X(X'X)⁻¹X' for design matrix X."""
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    return (U ** 2).sum(axis=1)  # h_ii = ||u_i||²


def _cooks_distance(X: np.ndarray, y: np.ndarray):
    """Cook's distance, hat-diagonal, and standardised residuals.

    Formula: D_i = (e_i² · h_i) / (p · MSE · (1 − h_i)²)
    """
    n, p = X.shape
    model = Ridge(alpha=1e-6).fit(X, y)
    y_pred = model.predict(X)
    residuals = y - y_pred
    mse = float(np.mean(residuals ** 2))
    h = _hat_diagonal(X)
    d = (residuals ** 2 * h) / (p * mse * (1 - h + 1e-10) ** 2)
    std_resid = residuals / (np.sqrt(mse) * np.sqrt(1 - h + 1e-10))
    return d, h, std_resid, residuals, model


# ── Figure 1: Regression Line Shifts ─────────────────────────────────────────

def plot_influence_concept(X_train: np.ndarray, y_train: np.ndarray,
                            feature_names: list[str]) -> go.Figure:
    """Show how three high-influence points shift the temp→cnt regression line."""
    temp_idx = feature_names.index("temp")
    x = X_train[:, temp_idx]
    y = y_train

    # Subsample for a cleaner scatter
    n_plot = min(1000, len(x))
    samp = RNG.choice(len(x), n_plot, replace=False)
    x_samp, y_samp = x[samp], y[samp]

    # Univariate ridge regression (with intercept via design matrix)
    X_uni = np.column_stack([np.ones(len(x)), x])   # (n, 2)
    _, h, std_resid, resid, model_full = _cooks_distance(X_uni, y)
    y_pred_full = model_full.predict(X_uni)

    # Cook's D
    mse = float(np.mean(resid ** 2))
    p = 2
    d = (resid ** 2 * h) / (p * mse * (1 - h + 1e-10) ** 2)

    # Top-3 influential instances (highest Cook's D)
    top3 = np.argsort(d)[-3:]

    # Refit without those 3 points
    keep = np.ones(len(x), dtype=bool)
    keep[top3] = False
    X_no3 = np.column_stack([np.ones(keep.sum()), x[keep]])
    model_no3 = Ridge(alpha=1e-6).fit(X_no3, y[keep])

    # Prediction curves
    x_range = np.linspace(x.min(), x.max(), 200)
    X_range = np.column_stack([np.ones(200), x_range])
    line_full = model_full.predict(X_range)
    line_no3 = model_no3.predict(X_range)

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[
            "Regression with all training instances",
            "Effect: removing the 3 high-influence points shifts the line",
        ],
        horizontal_spacing=0.12,
    )

    # ── Left panel ──────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=x_samp, y=y_samp,
        mode="markers",
        marker=dict(color=COLORS["primary"], size=4, opacity=0.20),
        name="Training data",
        hovertemplate="temp=%{x:.2f}<br>cnt=%{y:.0f}<extra></extra>",
    ), row=1, col=1)

    # Highlight influential points
    fig.add_trace(go.Scatter(
        x=x[top3], y=y[top3],
        mode="markers",
        marker=dict(color=COLORS["negative"], size=12, symbol="star",
                    line=dict(color="white", width=1)),
        name="High-influence (Cook's D top-3)",
        hovertemplate="temp=%{x:.2f}<br>cnt=%{y:.0f}<extra></extra>",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=x_range, y=line_full, mode="lines",
        line=dict(color=COLORS["accent"], width=2.5),
        name="Regression line (all data)",
    ), row=1, col=1)

    fig.update_xaxes(title_text="temp (normalised)", row=1, col=1)
    fig.update_yaxes(title_text="Bike rentals (cnt)", row=1, col=1)

    # ── Right panel ─────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=x_range, y=line_full, mode="lines",
        line=dict(color=COLORS["accent"], width=2.5),
        name="With influential points",
        showlegend=True,
    ), row=1, col=2)

    fig.add_trace(go.Scatter(
        x=x_range, y=line_no3, mode="lines",
        line=dict(color=COLORS["positive"], width=2.5, dash="dash"),
        name="Without influential points",
        showlegend=True,
    ), row=1, col=2)

    # Shade the difference
    fig.add_trace(go.Scatter(
        x=np.concatenate([x_range, x_range[::-1]]),
        y=np.concatenate([line_full, line_no3[::-1]]),
        fill="toself",
        fillcolor="rgba(219,39,119,0.10)",
        line=dict(color="rgba(0,0,0,0)"),
        showlegend=False,
        hoverinfo="skip",
    ), row=1, col=2)

    fig.add_annotation(
        x=0.5, y=0.5, xref="x2 domain", yref="y2 domain",
        text="← line shifts here",
        showarrow=True, arrowhead=2, ax=40, ay=0,
        font=dict(size=11, color=COLORS["neutral"]),
    )

    fig.update_xaxes(title_text="temp (normalised)", row=1, col=2)
    fig.update_yaxes(title_text="Bike rentals (cnt)", row=1, col=2)

    fig.update_layout(
        title=(
            "Influential Instances — Some Training Points Shape the Model More Than Others<br>"
            "<sup>Removing 3 high-influence points (star markers, high leverage + high residual) "
            "noticeably shifts the fitted regression line</sup>"
        ),
        height=500,
        legend=dict(orientation="h", y=-0.18),
    )
    return fig


# ── Figure 2: Cook's Distance Scatter ────────────────────────────────────────

def plot_cooks_concept(X_train: np.ndarray, y_train: np.ndarray) -> go.Figure:
    """Leverage vs |std residual| scatter, marker size = Cook's D."""
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X_train)

    # Subsample to keep hat-matrix computation tractable
    n_sub = 500
    idx_sub = RNG.choice(len(X_s), n_sub, replace=False)
    X_sub, y_sub = X_s[idx_sub], y_train[idx_sub]

    d, h, std_resid, _, _ = _cooks_distance(X_sub, y_sub)

    n, p = X_sub.shape
    thresh_d = 4 / n
    thresh_h = 2 * p / n
    influential = d > thresh_d

    # Marker sizes: scale Cook's D to a visible range [4, 22]
    size = 4 + 18 * (d - d.min()) / (d.max() - d.min() + 1e-10)

    colors = np.where(influential, COLORS["negative"], COLORS["primary"])

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=h[~influential], y=np.abs(std_resid[~influential]),
        mode="markers",
        marker=dict(color=COLORS["primary"], size=size[~influential], opacity=0.65),
        name="Typical instances",
        hovertemplate=(
            "Leverage=%{x:.4f}<br>|Std residual|=%{y:.2f}<extra></extra>"
        ),
    ))
    fig.add_trace(go.Scatter(
        x=h[influential], y=np.abs(std_resid[influential]),
        mode="markers",
        marker=dict(color=COLORS["negative"], size=size[influential],
                    opacity=0.85, line=dict(color="white", width=0.5)),
        name=f"Influential (Cook's D > 4/n = {thresh_d:.3f})",
        hovertemplate=(
            "Leverage=%{x:.4f}<br>|Std residual|=%{y:.2f}<extra></extra>"
        ),
    ))

    # Threshold lines
    fig.add_vline(x=thresh_h, line_dash="dash", line_color=COLORS["neutral"],
                  annotation_text=f"Leverage threshold 2p/n={thresh_h:.3f}",
                  annotation_position="top right",
                  annotation_font_size=11)
    fig.add_hline(y=2, line_dash="dash", line_color=COLORS["neutral"],
                  annotation_text="|Std resid| = 2 σ",
                  annotation_position="right",
                  annotation_font_size=11)

    # Quadrant labels
    for xr, yr, label in [
        (0.02, 0.15,  "Low leverage\nLow residual\n(safe)"),
        (0.02, 2.50,  "Low leverage\nHigh residual\n(outlier in y)"),
        (thresh_h + 0.002, 0.15,  "High leverage\nLow residual\n(safe)"),
        (thresh_h + 0.002, 2.50,  "High leverage\nHigh residual\n→ INFLUENTIAL"),
    ]:
        fig.add_annotation(
            x=xr, y=yr,
            text=label.replace("\n", "<br>"),
            showarrow=False,
            font=dict(size=10, color=COLORS["neutral"]),
            bgcolor="rgba(255,255,255,0.80)",
            bordercolor="#D1D5DB", borderwidth=1,
            align="left",
        )

    fig.update_layout(
        title=(
            "Cook's Distance — Points with High Leverage AND High Residual Are Influential<br>"
            "<sup>Marker size ∝ Cook's D = (e²·h) / (p·MSE·(1−h)²) — "
            "large red markers score high in both dimensions simultaneously</sup>"
        ),
        xaxis_title="Leverage (hat-matrix diagonal h_ii)",
        yaxis_title="|Standardised residual|",
        legend=dict(orientation="h", y=-0.18),
        height=520,
    )
    return fig


# ── Figure 3: TracIn Concept ──────────────────────────────────────────────────

def _fc_grads_batch(model, X_b: torch.Tensor,
                    y_b: torch.Tensor, device: str) -> torch.Tensor:
    """Closed-form per-instance fc-layer weight gradients (no backward needed).

    grad_i = delta_i^T ⊗ acts_i  →  flattened (10 × 1568,)
    """
    X_b, y_b = X_b.to(device), y_b.to(device)
    with torch.no_grad():
        h = F.relu(F.max_pool2d(model.conv1(X_b), 2))
        h = F.relu(F.max_pool2d(model.conv2(h), 2))
        acts = h.flatten(1)                               # (B, 1568)
        probs = F.softmax(model.fc(acts), dim=1)          # (B, 10)
    one_hot = torch.zeros_like(probs)
    one_hot.scatter_(1, y_b.unsqueeze(1), 1)
    delta = probs - one_hot                               # (B, 10)
    # outer product per instance → (B, 10, 1568) → (B, 10·1568)
    return torch.bmm(delta.unsqueeze(2), acts.unsqueeze(1)).reshape(len(X_b), -1)


def plot_tracin_concept(model, train_ds, test_ds, device: str = "cpu") -> go.Figure:
    """2×6 grid: col 1 = test digit-7 image, cols 2–6 = top/bottom TracIn training images."""
    targets_test = np.array(test_ds.targets)
    model.eval()

    # Find first correctly-classified digit 7 in test set
    test_img, test_pred = None, None
    for idx in np.where(targets_test == 7)[0]:
        img_t, lbl = test_ds[idx]
        with torch.no_grad():
            p = model(img_t.unsqueeze(0).to(device)).argmax(1).item()
        if p == int(lbl):
            test_img, test_pred = img_t, p
            break

    # Test-instance gradient
    y_test_t = torch.tensor([test_pred], device=device)
    test_grad = _fc_grads_batch(model, test_img.unsqueeze(0), y_test_t, device).squeeze()

    # Score 200 training instances spread across the training set
    n_train = len(train_ds)
    train_idx = list(range(0, n_train, n_train // 200))[:200]
    loader = DataLoader(Subset(train_ds, train_idx), batch_size=64, shuffle=False)

    scores, imgs, lbls = [], [], []
    for X_b, y_b in loader:
        batch_grads = _fc_grads_batch(model, X_b, y_b, device)   # (B, 10·1568)
        batch_scores = (batch_grads @ test_grad).cpu().numpy()    # (B,)
        scores.extend(batch_scores.tolist())
        imgs.extend(X_b.numpy())
        lbls.extend(y_b.numpy())

    scores = np.array(scores)
    imgs = np.array(imgs)
    lbls = np.array(lbls)

    top5_idx = np.argsort(scores)[-5:][::-1]     # highest TracIn score
    bot5_idx = np.argsort(scores)[:5]            # lowest TracIn score

    # Build subplot titles: 2 rows × 6 cols
    def _title(rank, score, label, kind):
        return f"#{rank+1} {kind}<br>score={score:+.3f} (lbl={label})"

    top_titles = (
        [f"Test image<br>(pred={test_pred})"] +
        [_title(i, scores[j], int(lbls[j]), "↑") for i, j in enumerate(top5_idx)]
    )
    bot_titles = (
        [""] +
        [_title(i, scores[j], int(lbls[j]), "↓") for i, j in enumerate(bot5_idx)]
    )

    fig = make_subplots(
        rows=2, cols=6,
        subplot_titles=top_titles + bot_titles,
        horizontal_spacing=0.03,
        vertical_spacing=0.18,
    )

    # Row 1: test image + top-5
    fig.add_trace(go.Heatmap(z=test_img[0].numpy(), colorscale="Gray_r",
                              showscale=False, hoverinfo="skip"), row=1, col=1)
    for col_i, tidx in enumerate(top5_idx, start=2):
        fig.add_trace(go.Heatmap(z=imgs[tidx].squeeze(), colorscale="Gray_r",
                                  showscale=False, hoverinfo="skip"),
                      row=1, col=col_i)

    # Row 2: test image (repeated for reference) + bottom-5
    fig.add_trace(go.Heatmap(z=test_img[0].numpy(), colorscale="Gray_r",
                              showscale=False, hoverinfo="skip"), row=2, col=1)
    for col_i, bidx in enumerate(bot5_idx, start=2):
        fig.add_trace(go.Heatmap(z=imgs[bidx].squeeze(), colorscale="Gray_r",
                                  showscale=False, hoverinfo="skip"),
                      row=2, col=col_i)

    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False, autorange="reversed")

    # Row labels
    for y_paper, row_label in [(0.80, "Most helpful<br>(high score)"),
                                (0.20, "Most harmful<br>(low score)")]:
        fig.add_annotation(
            x=-0.02, y=y_paper, xref="paper", yref="paper",
            text=f"<b>{row_label}</b>",
            showarrow=False, textangle=-90,
            font=dict(size=11, color=COLORS["neutral"]),
        )

    fig.update_layout(
        title=(
            "TracIn — Training Instances with Similar Gradients Are Most Influential<br>"
            "<sup>Score = dot product of per-instance fc-weight gradients with the test gradient. "
            "Positive = helped the test prediction; negative = hurt it.</sup>"
        ),
        height=480,
        margin=dict(l=70, t=100, b=50),
    )
    return fig


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    X_train, X_test, y_train, y_test, feature_names = load_regression()

    print("Figure 1: Influential instances / line shift…")
    fig1 = plot_influence_concept(X_train, y_train, feature_names)
    save_figure(fig1, CHAPTER, "how_influence_concept")

    print("Figure 2: Cook's distance scatter…")
    fig2 = plot_cooks_concept(X_train, y_train)
    save_figure(fig2, CHAPTER, "how_cooks_concept")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")

    train_ds, test_ds = load_images()
    print("Training SimpleCNN…")
    model = train_model(train_ds, epochs=3, device=device)
    model.eval()

    print("Figure 3: TracIn influential training images…")
    fig3 = plot_tracin_concept(model, train_ds, test_ds, device=device)
    save_figure(fig3, CHAPTER, "how_tracin_concept")


if __name__ == "__main__":
    main()

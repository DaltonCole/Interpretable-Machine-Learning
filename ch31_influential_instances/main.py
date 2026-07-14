"""
Chapter 31: Influential Instances
https://christophm.github.io/interpretable-ml-book/influential.html

Influential instances are training examples that disproportionately shape the
model's predictions. Methods: Cook's Distance (for linear models) and
TracInCP (gradient-based, via Captum) for neural networks.

Figures produced:
  1. cooks_distance.html/png   — Cook's distance for every training instance (linear model)
  2. influential_nn.html/png   — TracIn scores for a test instance (neural network)
"""
import numpy as np
import plotly.graph_objects as go
import torch
import torch.nn.functional as F
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Subset

from ch27_learned_features.main import SimpleCNN, train_model
from shared.datasets import load_images, load_regression
from shared.theme import COLORS, save_figure

N_SHOW_COOKS = 20   # highlight top-N outliers in Cook's distance plot


def cooks_distance(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute Cook's distance for each training instance."""
    n, p = X.shape
    model = LinearRegression().fit(X, y)
    y_pred = model.predict(X)
    residuals = y - y_pred
    mse = (residuals ** 2).mean()

    H = X @ np.linalg.pinv(X.T @ X) @ X.T  # hat matrix
    h = np.diag(H)
    d = (residuals ** 2 * h) / (p * mse * (1 - h + 1e-10) ** 2)
    return d


def plot_cooks_distance(d: np.ndarray, y: np.ndarray) -> go.Figure:
    threshold = 4 / len(d)
    influential = d > threshold

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=np.arange(len(d)),
        y=d,
        mode="markers",
        marker=dict(
            color=[COLORS["negative"] if inf else COLORS["primary"] for inf in influential],
            size=[8 if inf else 4 for inf in influential],
            opacity=0.7,
        ),
        hovertemplate="Instance %{x}<br>Cook's D = %{y:.4f}<extra></extra>",
        name="Cook's Distance",
    ))
    fig.add_hline(y=threshold, line_dash="dash", line_color=COLORS["neutral"],
                  annotation_text=f"threshold = 4/n = {threshold:.4f}")
    fig.update_layout(
        title="Cook's Distance — Influential Training Instances (Linear Model)",
        xaxis_title="Training instance index",
        yaxis_title="Cook's Distance",
    )
    return fig


def plot_influential_nn(top_imgs: np.ndarray, top_scores: np.ndarray,
                         top_labels: np.ndarray, test_img: np.ndarray,
                         test_pred: int) -> go.Figure:
    """Grid: test image (left) and most influential training images (right)."""
    from plotly.subplots import make_subplots

    n = len(top_imgs)
    fig = make_subplots(
        rows=1, cols=n + 1,
        subplot_titles=[f"Test (pred={test_pred})"] +
                        [f"Score:{s:.3f}<br>Label:{l}" for s, l in zip(top_scores, top_labels)],
    )
    fig.add_trace(go.Heatmap(z=test_img.squeeze(), colorscale="Gray_r",
                               showscale=False, hoverinfo="skip"), row=1, col=1)
    for i, img in enumerate(top_imgs):
        fig.add_trace(go.Heatmap(z=img.squeeze(), colorscale="Gray_r",
                                  showscale=False, hoverinfo="skip"), row=1, col=i + 2)
    fig.update_layout(
        title="Most Influential Training Images for a Test Prediction (TracIn)",
        height=220,
    )
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False, autorange="reversed")
    return fig


def main():
    # ── Cook's Distance on linear model (Bike Sharing) ────────────────────────
    X_train, X_test, y_train, y_test, feature_names = load_regression()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    print("Computing Cook's distances…")
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(len(X_scaled), 500, replace=False)
    d = cooks_distance(X_scaled[sample_idx], y_train[sample_idx])

    print(f"  {(d > 4/len(d)).sum()} influential instances (>{4/len(d):.4f})")
    fig1 = plot_cooks_distance(d, y_train[sample_idx])
    save_figure(fig1, "ch31_influential_instances", "cooks_distance")

    # ── TracIn on CNN (MNIST) ──────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")

    train_ds, test_ds = load_images()
    print("Training SimpleCNN…")
    model = train_model(train_ds, epochs=3, device=device)
    model.eval()

    print("Computing TracIn influence scores (per-instance fc-layer gradients)…")
    # Test instance: first digit 7 in test set
    test_targets = np.array(test_ds.targets)
    test_idx = np.where(test_targets == 7)[0][0]
    X_test_img, _ = test_ds[test_idx]
    X_test_t = X_test_img.unsqueeze(0).to(device)
    with torch.no_grad():
        test_pred = model(X_test_t).argmax(1).item()

    def fc_grads_batch(X_b: torch.Tensor, y_b: torch.Tensor) -> torch.Tensor:
        """Per-instance gradients of fc.weight via closed-form (no backward needed).

        grad_i = (softmax_i - one_hot_i)^T ⊗ activation_i  →  flattened (10 × hidden,)
        """
        with torch.no_grad():
            h = F.relu(F.max_pool2d(model.conv1(X_b), 2))
            h = F.relu(F.max_pool2d(model.conv2(h), 2))
            acts = h.flatten(1)           # (B, hidden)
            probs = F.softmax(model.fc(acts), dim=1)  # (B, 10)
        one_hot = torch.zeros_like(probs)
        one_hot.scatter_(1, y_b.unsqueeze(1), 1)
        delta = probs - one_hot           # (B, 10)
        # outer product per instance → (B, 10, hidden) → (B, 10*hidden)
        return torch.bmm(delta.unsqueeze(2), acts.unsqueeze(1)).reshape(len(X_b), -1)

    test_grad = fc_grads_batch(X_test_t, torch.tensor([test_pred]).to(device)).squeeze()

    # Score a subset of training instances
    train_subset = Subset(train_ds, list(range(2000)))
    loader = DataLoader(train_subset, batch_size=128, shuffle=False)
    scores, imgs, lbls = [], [], []
    for X_b, y_b in loader:
        X_b, y_b = X_b.to(device), y_b.to(device)
        batch_grads = fc_grads_batch(X_b, y_b)          # (B, 10*hidden)
        batch_scores = (batch_grads @ test_grad).cpu().numpy()  # (B,)
        scores.extend(batch_scores.tolist())
        imgs.extend(X_b.cpu().numpy())
        lbls.extend(y_b.cpu().numpy())

    scores = np.array(scores)
    imgs = np.array(imgs)
    lbls = np.array(lbls)
    top_idx = np.argsort(scores)[-8:][::-1]

    fig2 = plot_influential_nn(imgs[top_idx], scores[top_idx], lbls[top_idx],
                                X_test_img.numpy(), test_pred)
    save_figure(fig2, "ch31_influential_instances", "influential_nn")


if __name__ == "__main__":
    main()

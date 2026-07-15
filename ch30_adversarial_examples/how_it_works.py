"""
Chapter 30: How Adversarial Examples Work
https://christophm.github.io/interpretable-ml-book/adversarial.html

Conceptual figures showing how FGSM crafts imperceptible perturbations that
cross the model's decision boundary.

Figures produced:
  1. how_fgsm_concept.html/png           — original | sign(grad) | perturbation×10 | adversarial
  2. how_adversarial_imperceptible.html/png — 3×3 grid: original | adversarial | diff×20
  3. how_decision_boundary.html/png      — PCA pixel-space: digit 3 vs 5, FGSM arrows
"""
import numpy as np
import plotly.graph_objects as go
import torch
import torch.nn.functional as F
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, Subset

from ch27_learned_features.main import SimpleCNN, train_model
from shared.datasets import load_images
from shared.theme import COLORS, save_figure

CHAPTER = "ch30_adversarial_examples"
EPSILON = 0.2


# ── FGSM helper ───────────────────────────────────────────────────────────────

def fgsm(model, img_tensor: torch.Tensor, true_label: int,
          epsilon: float = EPSILON, device: str = "cpu") -> torch.Tensor:
    """x_adv = clamp(x + ε·sign(∇_x L(x,y)), 0, 1). Returns (1,28,28) tensor."""
    X = img_tensor.unsqueeze(0).float().to(device).detach().requires_grad_(True)
    loss = F.cross_entropy(model(X), torch.tensor([true_label], device=device))
    model.zero_grad()
    loss.backward()
    return (X + epsilon * X.grad.sign()).clamp(0.0, 1.0).detach().squeeze(0)


def _find_correct(model, test_ds, digit: int, device: str = "cpu"):
    """First correctly-classified test image of given digit.

    Returns (img_tensor (C,H,W), label, pred).
    """
    targets = np.array(test_ds.targets)
    model.eval()
    for idx in np.where(targets == digit)[0]:
        img_t, lbl = test_ds[idx]
        with torch.no_grad():
            p = model(img_t.unsqueeze(0).to(device)).argmax(1).item()
        if p == int(lbl):
            return img_t, int(lbl), p
    raise RuntimeError(f"No correctly-classified digit {digit} in test set")


# ── Figure 1: FGSM Concept ────────────────────────────────────────────────────

def plot_fgsm_concept(model, test_ds, device: str = "cpu") -> go.Figure:
    """4-panel: original | sign(gradient) | perturbation×10 | adversarial."""
    img_tensor, true_label, orig_pred = _find_correct(model, test_ds, 7, device)
    img_np = img_tensor[0].numpy()

    # Compute gradient sign and adversarial image
    X = img_tensor.unsqueeze(0).float().to(device).detach().requires_grad_(True)
    loss = F.cross_entropy(model(X), torch.tensor([true_label], device=device))
    model.zero_grad()
    loss.backward()
    grad_sign = X.grad[0, 0].detach().cpu().numpy()   # {-1, 0, 1}

    X_adv = (X + EPSILON * X.grad.sign()).clamp(0.0, 1.0).detach()
    with torch.no_grad():
        adv_pred = model(X_adv).argmax(1).item()

    adv_np = X_adv[0, 0].cpu().numpy()
    pert_np = EPSILON * grad_sign                       # actual perturbation

    attack_result = f"{orig_pred} → {adv_pred}" if adv_pred != orig_pred else f"{orig_pred} (unchanged)"

    fig = make_subplots(
        rows=1, cols=4,
        subplot_titles=[
            f"Original (pred={orig_pred})",
            "sign(∇ₓ Loss)",
            f"Perturbation ×10  (ε={EPSILON})",
            f"Adversarial (pred={adv_pred})",
        ],
        horizontal_spacing=0.05,
    )

    fig.add_trace(go.Heatmap(z=img_np, colorscale="Gray_r",
                              showscale=False, hoverinfo="skip"), row=1, col=1)
    fig.add_trace(go.Heatmap(z=grad_sign, colorscale="RdBu",
                              showscale=False, hoverinfo="skip"), row=1, col=2)
    fig.add_trace(go.Heatmap(z=pert_np * 10, colorscale="RdBu",
                              showscale=False, hoverinfo="skip"), row=1, col=3)
    fig.add_trace(go.Heatmap(z=adv_np, colorscale="Gray_r",
                              showscale=False, hoverinfo="skip"), row=1, col=4)

    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False, autorange="reversed")

    fig.update_layout(
        title=(
            "FGSM — Add the Sign of the Gradient to Fool the Model<br>"
            f"<sup>x_adv = x + ε·sign(∇_x L)  |  ε={EPSILON}  |  "
            f"Prediction: {attack_result}</sup>"
        ),
        height=360,
        margin=dict(t=100, b=40),
    )
    return fig


# ── Figure 2: Imperceptible Perturbations ─────────────────────────────────────

def plot_adversarial_imperceptible(model, test_ds, device: str = "cpu") -> go.Figure:
    """3×3 grid: for digits 1, 4, 8 show original | adversarial | difference×20."""
    digits_to_show = [1, 4, 8]
    rows_data = []
    for digit in digits_to_show:
        img_tensor, true_label, orig_pred = _find_correct(model, test_ds, digit, device)
        adv_tensor = fgsm(model, img_tensor, true_label, EPSILON, device)
        with torch.no_grad():
            adv_pred = model(adv_tensor.unsqueeze(0).to(device)).argmax(1).item()
        rows_data.append((
            img_tensor[0].numpy(),
            adv_tensor[0].cpu().numpy(),
            true_label, orig_pred, adv_pred,
        ))

    subplot_titles = []
    for _, _, true_label, orig_pred, adv_pred in rows_data:
        subplot_titles += [
            f"Original (true={true_label})",
            f"Adversarial (pred={adv_pred})",
            "Difference ×20",
        ]

    fig = make_subplots(
        rows=3, cols=3,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.05,
        vertical_spacing=0.10,
    )

    for row_i, (img_np, adv_np, true_label, orig_pred, adv_pred) in \
            enumerate(rows_data, start=1):
        diff = (adv_np - img_np) * 20
        fig.add_trace(go.Heatmap(z=img_np, colorscale="Gray_r",
                                  showscale=False, hoverinfo="skip"),
                      row=row_i, col=1)
        fig.add_trace(go.Heatmap(z=adv_np, colorscale="Gray_r",
                                  showscale=False, hoverinfo="skip"),
                      row=row_i, col=2)
        fig.add_trace(go.Heatmap(z=diff, colorscale="RdBu",
                                  showscale=False, hoverinfo="skip"),
                      row=row_i, col=3)

    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False, autorange="reversed")

    fig.update_layout(
        title=(
            "Adversarial Examples — Humans Can't See the Change, but the Model Can<br>"
            f"<sup>ε={EPSILON}: original and adversarial are visually identical; "
            "only the ×20-amplified difference reveals the structured perturbation</sup>"
        ),
        height=600,
        margin=dict(t=100, b=40),
    )
    return fig


# ── Figure 3: Decision Boundary ───────────────────────────────────────────────

def plot_decision_boundary(model, test_ds, device: str = "cpu") -> go.Figure:
    """PCA of flattened pixel values for digits 3 and 5; FGSM arrows show
    how small perturbations cross the decision boundary."""
    targets = np.array(test_ds.targets)

    # Collect correctly-classified images of digits 3 and 5
    def _collect(digit, max_n=80):
        imgs, idxs = [], []
        for idx in np.where(targets == digit)[0]:
            if len(imgs) >= max_n:
                break
            img_t, lbl = test_ds[idx]
            with torch.no_grad():
                p = model(img_t.unsqueeze(0).to(device)).argmax(1).item()
            if p == int(lbl):
                imgs.append(img_t[0].numpy().flatten())   # (784,)
                idxs.append(idx)
        return np.array(imgs), idxs

    model.eval()
    imgs_3, idxs_3 = _collect(3)
    imgs_5, _ = _collect(5)

    all_imgs = np.vstack([imgs_3, imgs_5])
    all_cls = np.array([3] * len(imgs_3) + [5] * len(imgs_5))

    pca = PCA(n_components=2, random_state=42)
    emb = pca.fit_transform(all_imgs)  # (N, 2)

    # Generate adversarial examples for digit 3, find ones that flip to 5
    arrows = []  # (orig_2d, adv_2d) pairs
    for idx in idxs_3:
        if len(arrows) >= 5:
            break
        img_t, lbl = test_ds[idx]
        adv_t = fgsm(model, img_t, int(lbl), EPSILON, device)
        with torch.no_grad():
            adv_pred = model(adv_t.unsqueeze(0).to(device)).argmax(1).item()
        if adv_pred != int(lbl):
            orig_2d = pca.transform(img_t[0].numpy().flatten().reshape(1, -1))[0]
            adv_2d = pca.transform(adv_t[0].cpu().numpy().flatten().reshape(1, -1))[0]
            arrows.append((orig_2d, adv_2d))

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=emb[:len(imgs_3), 0], y=emb[:len(imgs_3), 1],
        mode="markers",
        marker=dict(color=COLORS["primary"], size=7, opacity=0.55),
        name="Digit 3",
        hovertemplate="Digit 3<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=emb[len(imgs_3):, 0], y=emb[len(imgs_3):, 1],
        mode="markers",
        marker=dict(color=COLORS["accent"], size=7, opacity=0.55),
        name="Digit 5",
        hovertemplate="Digit 5<extra></extra>",
    ))

    # Draw FGSM perturbation arrows
    for orig_2d, adv_2d in arrows:
        fig.add_annotation(
            x=adv_2d[0], y=adv_2d[1],
            ax=orig_2d[0], ay=orig_2d[1],
            xref="x", yref="y", axref="x", ayref="y",
            arrowhead=3, arrowwidth=2,
            arrowcolor=COLORS["negative"],
            showarrow=True, text="",
        )

    # Mark adversarial endpoints
    if arrows:
        adv_pts = np.array([a[1] for a in arrows])
        fig.add_trace(go.Scatter(
            x=adv_pts[:, 0], y=adv_pts[:, 1],
            mode="markers",
            marker=dict(color=COLORS["negative"], size=10, symbol="star",
                        line=dict(color="white", width=1)),
            name=f"Adversarial 3→{arrows[0][1][0]:.0f} (misclassified)",
        ))

    n_arrows = len(arrows)
    fig.add_annotation(
        x=0.5, y=0.03, xref="paper", yref="paper",
        text=(
            f"Red arrows: FGSM (ε={EPSILON}) shifts {n_arrows} digit-3 images "
            "across the decision boundary (misclassified as digit 5)"
        ),
        showarrow=False,
        font=dict(size=11, color=COLORS["neutral"]),
        bgcolor="rgba(255,255,255,0.85)",
        bordercolor="#D1D5DB", borderwidth=1,
        xanchor="center",
    )

    fig.update_layout(
        title=(
            "Adversarial Examples — Tiny Perturbations Cross the Decision Boundary<br>"
            "<sup>PCA of raw 784-dim pixel vectors: digits 3 (blue) vs 5 (pink). "
            f"Short arrows (ε={EPSILON}) are enough to flip the prediction.</sup>"
        ),
        xaxis_title="PC1 (pixel space)",
        yaxis_title="PC2 (pixel space)",
        legend=dict(orientation="h", y=-0.18),
        height=520,
        margin=dict(b=90),
    )
    return fig


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    train_ds, test_ds = load_images()
    print("Training SimpleCNN…")
    model = train_model(train_ds, epochs=3, device=device)
    model.eval()

    print("Figure 1: FGSM concept…")
    fig1 = plot_fgsm_concept(model, test_ds, device=device)
    save_figure(fig1, CHAPTER, "how_fgsm_concept")

    print("Figure 2: Imperceptible perturbations…")
    fig2 = plot_adversarial_imperceptible(model, test_ds, device=device)
    save_figure(fig2, CHAPTER, "how_adversarial_imperceptible")

    print("Figure 3: Decision boundary in pixel-space PCA…")
    fig3 = plot_decision_boundary(model, test_ds, device=device)
    save_figure(fig3, CHAPTER, "how_decision_boundary")


if __name__ == "__main__":
    main()

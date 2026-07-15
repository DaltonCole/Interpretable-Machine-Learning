"""
Chapter 28: How Saliency Maps Work
https://christophm.github.io/interpretable-ml-book/pixel-wise.html

Conceptual figures explaining gradient-based saliency methods: vanilla gradient,
integrated gradients, and a side-by-side comparison across digit classes.

Figures produced:
  1. how_gradient_concept.html/png    — original | gradient | |grad| | grad×input
  2. how_ig_concept.html/png          — interpolation path and gradients at 3 α steps
  3. how_saliency_comparison.html/png — 4 digits × (original | vanilla | IG)
"""
import numpy as np
import plotly.graph_objects as go
import torch
import torch.nn.functional as F
from plotly.subplots import make_subplots

from ch27_learned_features.main import SimpleCNN, train_model
from shared.datasets import load_images
from shared.theme import COLORS, save_figure

CHAPTER = "ch28_saliency_maps"


def _normalize(arr: np.ndarray) -> np.ndarray:
    """Min-max normalise to [0, 1]."""
    mn, mx = arr.min(), arr.max()
    return (arr - mn) / (mx - mn + 1e-8)


def compute_vanilla_gradient(model, img_tensor: torch.Tensor,
                              target_class: int, device: str = "cpu") -> np.ndarray:
    """∂logit[target] / ∂pixel for one image. Returns (28, 28) array."""
    X = img_tensor.unsqueeze(0).float().to(device).detach().requires_grad_(True)
    logits = model(X)
    model.zero_grad()
    logits[0, target_class].backward()
    return X.grad[0, 0].detach().cpu().numpy()  # (28, 28)


def compute_integrated_gradients(model, img_tensor: torch.Tensor,
                                  target_class: int, n_steps: int = 20,
                                  device: str = "cpu") -> np.ndarray:
    """Integrated Gradients: (input − baseline) × avg gradient along α path.

    Returns (28, 28) attribution map.
    """
    baseline = torch.zeros_like(img_tensor)
    grads = []
    for k in range(n_steps + 1):
        alpha = k / n_steps
        interp = (baseline + alpha * img_tensor).unsqueeze(0).float().to(device)
        interp = interp.detach().requires_grad_(True)
        logits = model(interp)
        model.zero_grad()
        logits[0, target_class].backward()
        grads.append(interp.grad[0, 0].detach().cpu().numpy())
    avg_grad = np.mean(grads, axis=0)  # (28, 28)
    ig = (img_tensor[0].numpy() - baseline[0].numpy()) * avg_grad
    return ig


def _find_correct_digit(model, test_ds, digit: int,
                         device: str = "cpu"):
    """Return (img_tensor, label, pred) for the first correctly-classified instance."""
    targets = np.array(test_ds.targets)
    model.eval()
    for idx in np.where(targets == digit)[0]:
        img_t, lbl = test_ds[idx]
        with torch.no_grad():
            p = model(img_t.unsqueeze(0).to(device)).argmax(1).item()
        if p == int(lbl):
            return img_t, int(lbl), p
    raise RuntimeError(f"No correctly-classified digit {digit} found in test set")


# ── Figure 1: Gradient Concept ────────────────────────────────────────────────

def plot_gradient_concept(model, test_ds, device: str = "cpu") -> go.Figure:
    """4-panel: original | gradient | |gradient| | gradient × input."""
    # Prefer digit 3; fall back to 7
    for digit in [3, 7]:
        try:
            img_tensor, label, pred = _find_correct_digit(model, test_ds, digit, device)
            break
        except RuntimeError:
            continue

    img_np = img_tensor[0].numpy()  # (28, 28)
    grad_np = compute_vanilla_gradient(model, img_tensor, pred, device)
    abs_grad = np.abs(grad_np)
    grad_x_input = grad_np * img_np

    conf = float(F.softmax(
        model(img_tensor.unsqueeze(0).to(device)), dim=1
    )[0, pred].item())

    fig = make_subplots(
        rows=1, cols=4,
        subplot_titles=[
            "Original Image",
            "Gradient (∂logit / ∂pixel)",
            "|Gradient| (pixel importance)",
            "Gradient × Input",
        ],
        horizontal_spacing=0.05,
    )

    # col 1: original image
    fig.add_trace(go.Heatmap(z=img_np, colorscale="Gray_r",
                              showscale=False, hoverinfo="skip"), row=1, col=1)
    # col 2: raw gradient (signed — use RdBu: red=positive, blue=negative)
    fig.add_trace(go.Heatmap(z=_normalize(grad_np), colorscale="RdBu",
                              showscale=False, hoverinfo="skip"), row=1, col=2)
    # col 3: absolute gradient → pixel importance regardless of direction
    fig.add_trace(go.Heatmap(z=_normalize(abs_grad), colorscale="Hot",
                              showscale=False, hoverinfo="skip"), row=1, col=3)
    # col 4: gradient × input → mask out background pixels
    fig.add_trace(go.Heatmap(z=_normalize(grad_x_input), colorscale="Hot",
                              showscale=False, hoverinfo="skip"), row=1, col=4)

    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False, autorange="reversed")

    fig.update_layout(
        title=(
            "Saliency Maps — The Gradient Shows Which Pixels Matter Most<br>"
            f"<sup>Digit '{label}' correctly predicted as {pred} "
            f"(confidence {conf:.1%}) | "
            "Hot = high importance; grad×input masks out background</sup>"
        ),
        height=360,
        margin=dict(t=100, b=40),
    )
    return fig


# ── Figure 2: Integrated Gradients Concept ────────────────────────────────────

def plot_ig_concept(model, test_ds, device: str = "cpu") -> go.Figure:
    """2×3 grid showing 3 points on the IG path (top) and gradients there (bottom)."""
    img_tensor, label, pred = _find_correct_digit(model, test_ds, 3, device)

    baseline = torch.zeros_like(img_tensor)
    alphas = [0.0, 0.5, 1.0]
    col_titles_top = ["Baseline (α=0)", "Midpoint (α=0.5)", "Input (α=1.0)"]
    col_titles_bot = ["Gradient at α=0", "Gradient at α=0.5", "Gradient at α=1.0"]

    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=col_titles_top + col_titles_bot,
        horizontal_spacing=0.06,
        vertical_spacing=0.16,
    )

    for col_i, alpha in enumerate(alphas, start=1):
        interp = baseline + alpha * img_tensor                    # (1, 28, 28)
        interp_np = _normalize(interp[0].numpy())

        # Gradient at this alpha step
        X = interp.unsqueeze(0).float().to(device).detach().requires_grad_(True)
        logits = model(X)
        model.zero_grad()
        logits[0, pred].backward()
        grad_at_alpha = X.grad[0, 0].detach().cpu().numpy()      # (28, 28)

        fig.add_trace(go.Heatmap(z=interp_np, colorscale="Gray_r",
                                  showscale=False, hoverinfo="skip"),
                      row=1, col=col_i)
        fig.add_trace(go.Heatmap(z=_normalize(np.abs(grad_at_alpha)),
                                  colorscale="Hot",
                                  showscale=False, hoverinfo="skip"),
                      row=2, col=col_i)

    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False, autorange="reversed")

    # Row labels on the left
    for y_paper, label_text in [(0.78, "Input<br>at α"), (0.24, "Gradient<br>at α")]:
        fig.add_annotation(
            x=-0.04, y=y_paper, xref="paper", yref="paper",
            text=f"<b>{label_text}</b>",
            showarrow=False, textangle=-90,
            font=dict(size=11, color=COLORS["neutral"]),
        )

    fig.update_layout(
        title=(
            "Integrated Gradients — Average Gradients Along the Path from Baseline to Input<br>"
            "<sup>IG = (x − x₀) × ∫₀¹ ∂F(x₀ + α·(x−x₀)) / ∂x dα  |  "
            "At α=0 (black image) gradients are ~0; they grow as the image emerges</sup>"
        ),
        height=500,
        margin=dict(l=70, t=100, b=50),
    )
    return fig


# ── Figure 3: Saliency Comparison ─────────────────────────────────────────────

def plot_saliency_comparison(model, test_ds, device: str = "cpu") -> go.Figure:
    """4×3 grid: 4 digit classes × (original | vanilla |grad| | IG)."""
    digits_to_show = [0, 3, 6, 9]
    rows_data = []
    for digit in digits_to_show:
        img_tensor, label, pred = _find_correct_digit(model, test_ds, digit, device)
        grad = compute_vanilla_gradient(model, img_tensor, pred, device)
        ig = compute_integrated_gradients(model, img_tensor, pred,
                                           n_steps=20, device=device)
        rows_data.append((img_tensor[0].numpy(), grad, ig, label, pred))

    # subplot_titles: 4 rows × 3 cols = 12 entries (fill left-to-right)
    subplot_titles = []
    for _, _, _, label, pred in rows_data:
        subplot_titles += [f"Digit {label}", "Vanilla |grad|", "Integrated Gradients"]

    fig = make_subplots(
        rows=4, cols=3,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.04,
        vertical_spacing=0.07,
    )

    for row_i, (img_np, grad, ig, label, pred) in enumerate(rows_data, start=1):
        fig.add_trace(go.Heatmap(z=img_np, colorscale="Gray_r",
                                  showscale=False, hoverinfo="skip"),
                      row=row_i, col=1)
        fig.add_trace(go.Heatmap(z=_normalize(np.abs(grad)), colorscale="Hot",
                                  showscale=False, hoverinfo="skip"),
                      row=row_i, col=2)
        fig.add_trace(go.Heatmap(z=_normalize(np.abs(ig)), colorscale="Hot",
                                  showscale=False, hoverinfo="skip"),
                      row=row_i, col=3)

    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False, autorange="reversed")

    fig.update_layout(
        title=(
            "Saliency Methods — Vanilla Gradient vs Integrated Gradients<br>"
            "<sup>Hot = high importance. IG accumulates attributions along the "
            "interpolation path, producing sharper and less noisy maps.</sup>"
        ),
        height=700,
        margin=dict(t=100, b=40),
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

    print("Figure 1: Gradient concept…")
    fig1 = plot_gradient_concept(model, test_ds, device=device)
    save_figure(fig1, CHAPTER, "how_gradient_concept")

    print("Figure 2: Integrated gradients concept…")
    fig2 = plot_ig_concept(model, test_ds, device=device)
    save_figure(fig2, CHAPTER, "how_ig_concept")

    print("Figure 3: Saliency comparison (4 digits × 3 methods)…")
    fig3 = plot_saliency_comparison(model, test_ds, device=device)
    save_figure(fig3, CHAPTER, "how_saliency_comparison")


if __name__ == "__main__":
    main()

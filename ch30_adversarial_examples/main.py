"""
Chapter 30: Adversarial Examples
https://christophm.github.io/interpretable-ml-book/adversarial.html

Adversarial examples are inputs perturbed by a tiny, imperceptible amount
that cause the model to misclassify. This chapter uses FGSM (Fast Gradient
Sign Method) to generate adversarial MNIST images.

Figures produced:
  1. adversarial_grid.html/png    — original / perturbation / adversarial grid
  2. confidence_vs_epsilon.html/png — attack success rate and confidence vs. ε
"""
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import torch
import torch.nn.functional as F
from plotly.subplots import make_subplots
from torch.utils.data import DataLoader, Subset

from ch27_learned_features.main import SimpleCNN, train_model
from shared.datasets import load_images
from shared.theme import COLORS, save_figure


def fgsm_attack(model, X: torch.Tensor, y: torch.Tensor,
                epsilon: float) -> torch.Tensor:
    """Fast Gradient Sign Method: perturb X toward the gradient of the loss."""
    X_adv = X.clone().requires_grad_(True)
    loss = F.cross_entropy(model(X_adv), y)
    loss.backward()
    return (X_adv + epsilon * X_adv.grad.sign()).clamp(0, 1).detach()


def plot_adversarial_grid(originals, perturbations, adversarials,
                           true_labels, orig_preds, adv_preds,
                           epsilon: float) -> go.Figure:
    n = len(originals)
    fig = make_subplots(
        rows=3, cols=n,
        row_titles=["Original", f"Perturbation ×10 (ε={epsilon})", "Adversarial"],
        subplot_titles=(
            [f"True:{l} Pred:{p}" for l, p in zip(true_labels, orig_preds)] +
            [""] * n +
            [f"Pred:{p}" for p in adv_preds]
        ),
    )
    for i in range(n):
        orig = originals[i].squeeze()
        pert = perturbations[i].squeeze() * 10  # amplify for visibility
        adv = adversarials[i].squeeze()
        for row, img, cscale in [(1, orig, "Gray_r"), (2, pert, "RdBu"), (3, adv, "Gray_r")]:
            fig.add_trace(go.Heatmap(z=img, colorscale=cscale, showscale=False, hoverinfo="skip"),
                          row=row, col=i + 1)
    fig.update_layout(title=f"FGSM Adversarial Examples (ε={epsilon})", height=520)
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False, autorange="reversed")
    return fig


def plot_epsilon_sweep(epsilons, success_rates, mean_confidences) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=epsilons, y=success_rates, mode="lines+markers",
        name="Attack success rate",
        line=dict(color=COLORS["negative"], width=2),
        yaxis="y",
    ))
    fig.add_trace(go.Scatter(
        x=epsilons, y=mean_confidences, mode="lines+markers",
        name="Mean adversarial confidence",
        line=dict(color=COLORS["accent"], width=2, dash="dash"),
        yaxis="y2",
    ))
    fig.update_layout(
        title="FGSM Attack: Success Rate and Confidence vs. Epsilon",
        xaxis_title="Epsilon (perturbation magnitude)",
        yaxis=dict(title="Attack success rate", range=[0, 1], tickformat=".0%"),
        yaxis2=dict(title="Mean confidence", range=[0, 1], overlaying="y", side="right"),
        legend=dict(x=0.05, y=0.95),
    )
    return fig


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    train_ds, test_ds = load_images()
    print("Training SimpleCNN…")
    model = train_model(train_ds, epochs=3, device=device)
    model.eval()

    # Select one image per digit 0-5
    targets = np.array(test_ds.targets)
    sel_idx = [np.where(targets == d)[0][0] for d in range(6)]
    subset = Subset(test_ds, sel_idx)
    loader = DataLoader(subset, batch_size=6)
    X_sel, y_sel = next(iter(loader))
    X_sel, y_sel = X_sel.to(device), y_sel.to(device)

    epsilon = 0.2
    print(f"Generating adversarial examples (ε={epsilon})…")
    X_adv = fgsm_attack(model, X_sel, y_sel, epsilon)
    perturbation = X_adv - X_sel

    with torch.no_grad():
        orig_preds = model(X_sel).argmax(1).cpu().numpy()
        adv_preds = model(X_adv).argmax(1).cpu().numpy()

    fig1 = plot_adversarial_grid(
        X_sel.cpu().numpy(), perturbation.cpu().numpy(), X_adv.cpu().numpy(),
        y_sel.cpu().numpy(), orig_preds, adv_preds, epsilon,
    )
    save_figure(fig1, "ch30_adversarial_examples", "adversarial_grid")

    print("Sweeping epsilon values…")
    epsilons = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
    loader_full = DataLoader(Subset(test_ds, list(range(500))), batch_size=128)
    success_rates, mean_confidences = [], []
    for eps in epsilons:
        n_success, total, conf_sum = 0, 0, 0
        for X_b, y_b in loader_full:
            X_b, y_b = X_b.to(device), y_b.to(device)
            X_b_adv = fgsm_attack(model, X_b, y_b, eps)
            with torch.no_grad():
                logits = model(X_b_adv)
                probs = F.softmax(logits, dim=1)
                adv_p = logits.argmax(1)
                n_success += (adv_p != y_b).sum().item()
                conf_sum += probs.max(1).values.sum().item()
                total += len(y_b)
        success_rates.append(n_success / total)
        mean_confidences.append(conf_sum / total)
        print(f"  ε={eps:.2f}  success={n_success/total:.2%}  conf={conf_sum/total:.3f}")

    fig2 = plot_epsilon_sweep(epsilons, success_rates, mean_confidences)
    save_figure(fig2, "ch30_adversarial_examples", "confidence_vs_epsilon")


if __name__ == "__main__":
    main()

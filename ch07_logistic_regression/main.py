"""
Chapter 7: Logistic Regression
https://christophm.github.io/interpretable-ml-book/logistic.html

Figures produced:
  1. odds_ratios.html/png   — log-odds coefficients with 95% CI
  2. probability_curve.html/png — predicted probability vs. a single feature (others at mean)
"""
import numpy as np
import plotly.graph_objects as go
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from shared.datasets import load_classification
from shared.theme import COLORS, save_figure


def plot_odds_ratios(coefs: np.ndarray, ci: np.ndarray, feature_names: list) -> go.Figure:
    """Log-odds coefficient plot sorted by magnitude."""
    order = np.argsort(np.abs(coefs))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=coefs[order],
        y=[feature_names[i] for i in order],
        mode="markers",
        error_x=dict(type="data", array=ci[order], visible=True, color=COLORS["neutral"]),
        marker=dict(size=10, color=COLORS["primary"]),
    ))
    fig.add_vline(x=0, line_dash="dash", line_color=COLORS["neutral"], line_width=1)
    fig.update_layout(
        title="Logistic Regression — Log-Odds Coefficients (95% CI)",
        xaxis_title="Log-Odds",
        yaxis_title="Feature",
        height=600,
    )
    return fig


def plot_probability_curve(model, scaler, X_train, feature_names, feature_idx: int) -> go.Figure:
    """Predicted probability as one feature varies; all others held at their mean."""
    X_mean = np.mean(X_train, axis=0)
    feature_range = np.linspace(X_train[:, feature_idx].min(), X_train[:, feature_idx].max(), 200)

    X_sweep = np.tile(X_mean, (200, 1))
    X_sweep[:, feature_idx] = feature_range

    probs = model.predict_proba(scaler.transform(X_sweep))[:, 1]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=feature_range,
        y=probs,
        mode="lines",
        line=dict(color=COLORS["primary"], width=3),
        name="P(malignant)",
    ))
    fig.add_hline(y=0.5, line_dash="dash", line_color=COLORS["neutral"])
    fig.update_layout(
        title=f"Predicted Probability vs. {feature_names[feature_idx]}",
        xaxis_title=feature_names[feature_idx],
        yaxis_title="P(malignant)",
        yaxis=dict(range=[0, 1]),
    )
    return fig


def main():
    X_train, X_test, y_train, y_test, feature_names, class_names = load_classification()

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_s, y_train)

    coefs = model.coef_[0]
    # Approximate 95% CI using Wald method
    n = X_train_s.shape[0]
    p_hat = model.predict_proba(X_train_s)[:, 1]
    W = np.diag(p_hat * (1 - p_hat))
    fisher_info = X_train_s.T @ W @ X_train_s
    se = np.sqrt(np.diag(np.linalg.pinv(fisher_info)))
    ci_95 = stats.norm.ppf(0.975) * se

    print("Generating odds-ratio plot…")
    fig1 = plot_odds_ratios(coefs, ci_95, feature_names)
    save_figure(fig1, "ch07_logistic_regression", "odds_ratios")

    # Use the feature with the largest absolute coefficient for the curve
    top_feature_idx = int(np.argmax(np.abs(coefs)))
    print(f"Generating probability curve for '{feature_names[top_feature_idx]}'…")
    fig2 = plot_probability_curve(model, scaler, X_train, feature_names, top_feature_idx)
    save_figure(fig2, "ch07_logistic_regression", "probability_curve")


if __name__ == "__main__":
    main()

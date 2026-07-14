"""
Chapter 24: Leave One Feature Out (LOFO) Importance
https://christophm.github.io/interpretable-ml-book/lofo.html

LOFO importance trains the model without each feature and measures the
performance drop — more robust to correlated features than permutation importance.

Figures produced:
  1. lofo_importance.html/png — LOFO importance ± std across CV folds
  2. lofo_vs_perm.html/png    — LOFO vs. permutation importance scatter

Requires: lofo-importance  (pip install lofo-importance)
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from shared.datasets import load_regression
from shared.models import get_regression_model
from shared.theme import COLORS, save_figure


def plot_lofo_importance(importance_df: pd.DataFrame) -> go.Figure:
    df = importance_df.sort_values("importance_mean")
    fig = go.Figure(go.Bar(
        x=df["importance_mean"].values,
        y=df["feature"].values,
        orientation="h",
        error_x=dict(type="data", array=df["importance_std"].values, visible=True),
        marker_color=[COLORS["primary"] if v > 0 else COLORS["negative"]
                      for v in df["importance_mean"].values],
    ))
    fig.add_vline(x=0, line_color=COLORS["neutral"], line_width=1,
                  line_dash="dash", annotation_text="no change")
    fig.update_layout(
        title="LOFO Feature Importance (CV-validated)",
        xaxis_title="Mean decrease in OOF R²",
    )
    return fig


def plot_lofo_vs_perm(lofo_df: pd.DataFrame, perm_means: np.ndarray,
                      feature_names: list) -> go.Figure:
    """Scatter comparing LOFO and permutation importance per feature."""
    perm_map = {n: v for n, v in zip(feature_names, perm_means)}
    merged = lofo_df.copy()
    merged["perm"] = merged["feature"].map(perm_map)

    fig = px.scatter(
        merged,
        x="perm",
        y="importance_mean",
        text="feature",
        labels={"perm": "Permutation Importance (ΔR²)",
                "importance_mean": "LOFO Importance (ΔR²)"},
        title="LOFO vs. Permutation Importance",
        color_discrete_sequence=[COLORS["primary"]],
    )
    fig.update_traces(textposition="top center", marker_size=10)
    return fig


def main():
    X_train, X_test, y_train, y_test, feature_names = load_regression()

    try:
        import lofo
        train_df = pd.DataFrame(X_train, columns=feature_names)
        train_df["cnt"] = y_train

        dataset = lofo.Dataset(train_df, target="cnt", features=feature_names)
        lofo_imp = lofo.LOFOImportance(dataset, scoring="r2")
        importance_df = lofo_imp.get_importance()
        importance_df = importance_df.rename(columns={
            "feature": "feature",
            "importance_mean": "importance_mean",
            "importance_std": "importance_std",
        })
    except ImportError:
        print("lofo-importance not installed — using placeholder data.")
        print("Install with: pip install lofo-importance")
        model = get_regression_model(X_train, y_train)
        # Fallback: approximate LOFO by retraining without each feature
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import r2_score
        base_score = r2_score(y_test, model.predict(X_test))
        rows = []
        for i, name in enumerate(feature_names):
            cols = [j for j in range(X_train.shape[1]) if j != i]
            m = RandomForestRegressor(n_estimators=50, random_state=42)
            m.fit(X_train[:, cols], y_train)
            score = r2_score(y_test, m.predict(X_test[:, cols]))
            rows.append({"feature": name, "importance_mean": base_score - score, "importance_std": 0.0})
        importance_df = pd.DataFrame(rows)

    print("Generating LOFO importance chart…")
    fig1 = plot_lofo_importance(importance_df)
    save_figure(fig1, "ch24_lofo", "lofo_importance")

    print("Generating LOFO vs. permutation comparison…")
    from sklearn.inspection import permutation_importance
    model = get_regression_model(X_train, y_train)
    perm_result = permutation_importance(model, X_test, y_test, n_repeats=5,
                                          random_state=42, scoring="r2")
    fig2 = plot_lofo_vs_perm(importance_df, perm_result.importances_mean, feature_names)
    save_figure(fig2, "ch24_lofo", "lofo_vs_perm")


if __name__ == "__main__":
    main()

"""
Chapter 15: Counterfactual Explanations
https://christophm.github.io/interpretable-ml-book/counterfactual.html

A counterfactual answers: "What is the minimal change to the input that
flips the model's prediction?"

Figures produced:
  1. counterfactual_comparison.html/png — original vs. counterfactual feature values
  2. counterfactual_changes.html/png    — delta (change) per feature

Uses: dice-ml (Diverse Counterfactual Explanations)
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from shared.datasets import load_classification, load_regression, load_regression_df
from shared.models import get_regression_model
from shared.theme import COLORS, save_figure


def plot_comparison(original: pd.Series, counterfactual: pd.Series,
                    original_pred: float, cf_pred: float) -> go.Figure:
    """Side-by-side bar chart comparing original and counterfactual feature values."""
    features = original.index.tolist()
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name=f"Original (pred={original_pred:.0f})",
        x=features,
        y=original.values,
        marker_color=COLORS["primary"],
    ))
    fig.add_trace(go.Bar(
        name=f"Counterfactual (pred={cf_pred:.0f})",
        x=features,
        y=counterfactual.values,
        marker_color=COLORS["accent"],
    ))
    fig.update_layout(
        barmode="group",
        title="Original vs. Counterfactual — Feature Values",
        xaxis_title="Feature",
        yaxis_title="Value",
        xaxis_tickangle=-35,
    )
    return fig


def plot_changes(delta: pd.Series) -> go.Figure:
    """Horizontal bar of feature changes (only changed features shown)."""
    changed = delta[delta.abs() > 1e-6].sort_values()
    colors = [COLORS["positive"] if v > 0 else COLORS["negative"] for v in changed.values]
    fig = go.Figure(go.Bar(
        x=changed.values,
        y=changed.index.tolist(),
        orientation="h",
        marker_color=colors,
    ))
    fig.add_vline(x=0, line_color=COLORS["neutral"], line_width=1)
    fig.update_layout(
        title="Counterfactual — Changes Required",
        xaxis_title="Change in feature value",
        yaxis_title="Feature",
    )
    return fig


def main():
    import dice_ml
    from dice_ml import Dice

    X_train, X_test, y_train, y_test, feature_names = load_regression()

    model = get_regression_model(X_train, y_train)

    # Wrap in pandas for DiCE
    train_df = pd.DataFrame(X_train, columns=feature_names)
    train_df["cnt"] = y_train
    test_df = pd.DataFrame(X_test, columns=feature_names)

    dice_data = dice_ml.Data(
        dataframe=train_df,
        continuous_features=feature_names,
        outcome_name="cnt",
    )
    dice_model = dice_ml.Model(model=model, backend="sklearn", model_type="regressor")
    exp = Dice(dice_data, dice_model, method="random")

    instance = test_df.iloc[[0]]
    original_pred = model.predict(instance.values)[0]
    print(f"Original prediction: {original_pred:.0f}")

    # Request a counterfactual that increases rental count by ~200
    desired_range = [original_pred + 150, original_pred + 250]
    print(f"Seeking counterfactual in range {desired_range}…")
    cf = exp.generate_counterfactuals(
        instance, total_CFs=1, desired_range=desired_range
    )

    cf_df = cf.cf_examples_list[0].final_cfs_df
    cf_instance = cf_df.iloc[0][feature_names]
    cf_pred = model.predict(cf_instance.values.reshape(1, -1))[0]
    print(f"Counterfactual prediction: {cf_pred:.0f}")

    original_series = instance.iloc[0][feature_names]
    delta = cf_instance - original_series

    print("Generating comparison plot…")
    fig1 = plot_comparison(original_series, cf_instance, original_pred, cf_pred)
    save_figure(fig1, "ch15_counterfactual", "counterfactual_comparison")

    print("Generating changes plot…")
    fig2 = plot_changes(delta)
    save_figure(fig2, "ch15_counterfactual", "counterfactual_changes")


if __name__ == "__main__":
    main()

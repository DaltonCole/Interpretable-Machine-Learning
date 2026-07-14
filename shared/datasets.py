"""
Shared dataset loaders used across all chapters.

Primary datasets:
  - Bike Sharing (UCI #275): hourly bike rental counts — used for regression
  - Breast Cancer (sklearn): binary tumor classification
  - MNIST (torchvision): handwritten digits — used for neural network chapters
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
DATA_DIR = Path(__file__).parent.parent / "data"


def load_regression():
    """Bike Sharing Dataset (UCI #275) — hourly counts.

    Returns X_train, X_test, y_train, y_test, feature_names.
    Target: cnt (total hourly rentals).
    """
    from ucimlrepo import fetch_ucirepo

    dataset = fetch_ucirepo(id=275)
    X: pd.DataFrame = dataset.data.features.copy()
    y: pd.Series = dataset.data.targets["cnt"]

    # Drop non-predictive or leaky columns
    X = X.drop(columns=[c for c in ["instant", "dteday"] if c in X.columns])

    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y.values, test_size=0.2, random_state=RANDOM_STATE
    )
    return X_train, X_test, y_train.astype(float), y_test.astype(float), list(X.columns)


def load_regression_df():
    """Same as load_regression() but returns raw DataFrames before splitting.

    Useful for chapters that need feature names alongside values (e.g. SHAP, LIME).
    Returns X (DataFrame), y (Series).
    """
    from ucimlrepo import fetch_ucirepo

    dataset = fetch_ucirepo(id=275)
    X: pd.DataFrame = dataset.data.features.copy()
    y: pd.Series = dataset.data.targets["cnt"]
    X = X.drop(columns=[c for c in ["instant", "dteday"] if c in X.columns])
    return X, y


def load_classification():
    """Breast Cancer Wisconsin dataset — binary classification.

    Returns X_train, X_test, y_train, y_test, feature_names, class_names.
    """
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=RANDOM_STATE
    )
    return (
        X_train,
        X_test,
        y_train,
        y_test,
        list(data.feature_names),
        list(data.target_names),
    )


def load_images():
    """MNIST dataset for neural network interpretation chapters.

    Returns (train_dataset, test_dataset) as torchvision Dataset objects.
    """
    from torchvision import datasets, transforms

    mnist_dir = DATA_DIR / "mnist"
    mnist_dir.mkdir(parents=True, exist_ok=True)

    transform = transforms.Compose([transforms.ToTensor()])
    train = datasets.MNIST(root=str(mnist_dir), train=True, download=True, transform=transform)
    test = datasets.MNIST(root=str(mnist_dir), train=False, download=True, transform=transform)
    return train, test

"""
Pre-trained model helpers reused across chapters.

All models are trained on the shared datasets with a fixed random seed so results
are reproducible and comparable across chapters.
"""
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

RANDOM_STATE = 42


def get_regression_model(X_train, y_train, model_type: str = "random_forest"):
    """Return a fitted regression model.

    model_type options: "random_forest", "linear", "ridge", "decision_tree", "gradient_boosting"
    """
    models = {
        "random_forest": RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE),
        "linear": Pipeline([("scaler", StandardScaler()), ("lr", LinearRegression())]),
        "ridge": Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))]),
        "decision_tree": DecisionTreeRegressor(max_depth=5, random_state=RANDOM_STATE),
        "gradient_boosting": GradientBoostingRegressor(n_estimators=100, random_state=RANDOM_STATE),
    }
    model = models[model_type]
    model.fit(X_train, y_train)
    return model


def get_classification_model(X_train, y_train, model_type: str = "random_forest"):
    """Return a fitted classification model.

    model_type options: "random_forest", "logistic", "decision_tree"
    """
    models = {
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
        "logistic": Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ]),
        "decision_tree": DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE),
    }
    model = models[model_type]
    model.fit(X_train, y_train)
    return model

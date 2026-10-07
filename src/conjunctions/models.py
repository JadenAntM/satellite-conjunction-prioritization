"""Fixed tabular baselines with training-fold preprocessing."""

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from .features import feature_columns


def signed_log(values):
    return np.sign(values) * np.log1p(np.abs(values))


def make_model(family, information):
    columns = feature_columns(information)
    categorical = [c for c in columns if c == "latest__c_object_type"]
    numeric = [c for c in columns if c not in categorical]
    plain = [c for c in numeric if "__risk" in c]
    logged = [c for c in numeric if c not in plain]
    blocks = []
    for name, fields, transform in [("plain", plain, False), ("logged", logged, True)]:
        if not fields:
            continue
        steps = []
        if transform:
            steps.append(("log", FunctionTransformer(signed_log, feature_names_out="one-to-one")))
        steps.append(("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)))
        if family == "logistic":
            steps.append(("scale", StandardScaler()))
        blocks.append((name, Pipeline(steps), fields))
    if categorical:
        blocks.append(("categorical", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), categorical))
    if family == "logistic":
        estimator = LogisticRegression(C=1, max_iter=3000, random_state=42)
    elif family == "random_forest":
        estimator = RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_leaf=10, n_jobs=2, random_state=42)
    elif family == "hgb":
        estimator = HistGradientBoostingClassifier(max_iter=150, learning_rate=.05, max_leaf_nodes=15,
                                                   l2_regularization=1, early_stopping=False, random_state=42)
    else:
        raise ValueError(f"Unknown model family: {family}")
    return Pipeline([("features", ColumnTransformer(blocks, remainder="drop")), ("estimator", estimator)])

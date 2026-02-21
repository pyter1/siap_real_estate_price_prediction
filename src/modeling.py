from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from .evaluation import evaluate_regression
from .features import FeatureSpec, build_preprocess_pipeline, infer_feature_spec, split_xy


@dataclass
class TrainArtifacts:
    rf_pipeline: Pipeline
    xgb_pipeline: Pipeline
    rf_metrics: Dict[str, float]
    xgb_metrics: Dict[str, float]


def _stratify_bins(y: pd.Series, bins: int = 10) -> pd.Series:
    try:
        return pd.qcut(y, q=bins, duplicates="drop")
    except Exception:
        return pd.Series([0] * len(y))


def train_models(
    df: pd.DataFrame,
    *,
    target_col: str,
    random_seed: int,
    test_size: float,
    cv_folds: int,
    n_iter_search: int,
) -> Tuple[TrainArtifacts, Dict[str, Any]]:
    X, y = split_xy(df, target_col=target_col)

    strat = _stratify_bins(y, bins=10)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_seed,
        stratify=strat,
    )

    spec: FeatureSpec = infer_feature_spec(df)
    preprocess = build_preprocess_pipeline(spec)

    rf = RandomForestRegressor(
        random_state=random_seed,
        n_estimators=400,
        n_jobs=-1,
    )
    rf_pipe = Pipeline(steps=[("preprocess", preprocess), ("model", rf)])

    rf_param_dist = {
        "model__n_estimators": [300, 500, 800],
        "model__max_depth": [None, 8, 12, 18, 24],
        "model__min_samples_split": [2, 5, 10, 20],
        "model__min_samples_leaf": [1, 2, 4, 8],
        "model__max_features": ["sqrt", 0.6, 0.8, 1.0],
    }

    rf_search = RandomizedSearchCV(
        estimator=rf_pipe,
        param_distributions=rf_param_dist,
        n_iter=n_iter_search,
        scoring="neg_mean_absolute_error",
        cv=cv_folds,
        random_state=random_seed,
        n_jobs=-1,
        verbose=0,
    )
    rf_search.fit(X_train, y_train)
    rf_best = rf_search.best_estimator_

    xgb = XGBRegressor(
        objective="reg:squarederror",
        random_state=random_seed,
        n_estimators=1200,
        n_jobs=-1,
        tree_method="hist",
    )
    xgb_pipe = Pipeline(steps=[("preprocess", preprocess), ("model", xgb)])

    xgb_param_dist = {
        "model__n_estimators": [600, 900, 1200, 1600],
        "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
        "model__max_depth": [3, 4, 5, 6, 8],
        "model__subsample": [0.7, 0.85, 1.0],
        "model__colsample_bytree": [0.7, 0.85, 1.0],
        "model__min_child_weight": [1, 3, 5, 7],
        "model__reg_alpha": [0.0, 0.5, 1.0],
        "model__reg_lambda": [1.0, 2.0, 5.0],
    }

    xgb_search = RandomizedSearchCV(
        estimator=xgb_pipe,
        param_distributions=xgb_param_dist,
        n_iter=n_iter_search,
        scoring="neg_mean_absolute_error",
        cv=cv_folds,
        random_state=random_seed,
        n_jobs=-1,
        verbose=0,
    )
    xgb_search.fit(X_train, y_train)
    xgb_best = xgb_search.best_estimator_

    rf_pred = rf_best.predict(X_test)
    xgb_pred = xgb_best.predict(X_test)

    rf_metrics = evaluate_regression(y_test.values, rf_pred)
    xgb_metrics = evaluate_regression(y_test.values, xgb_pred)

    debug = {
        "feature_spec": spec.__dict__,
        "rf_best_params": rf_search.best_params_,
        "xgb_best_params": xgb_search.best_params_,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
    }

    return TrainArtifacts(
        rf_pipeline=rf_best,
        xgb_pipeline=xgb_best,
        rf_metrics=rf_metrics,
        xgb_metrics=xgb_metrics,
    ), debug


def save_pipeline(pipeline: Pipeline, path: str) -> None:
    joblib.dump(pipeline, path)


def load_pipeline(path: str) -> Pipeline:
    return joblib.load(path)

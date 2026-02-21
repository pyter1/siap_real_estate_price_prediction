from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd

try:
    from category_encoders.target_encoder import TargetEncoder
except Exception:  # pragma: no cover
    TargetEncoder = None  # type: ignore

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


@dataclass(frozen=True)
class FeatureSpec:
    numeric: List[str]
    categorical_low_card: List[str]
    categorical_high_card: List[str]


def infer_feature_spec(df: pd.DataFrame) -> FeatureSpec:
    numeric = []
    for c in [
        "area_m2", "rooms", "floor", "lift", "parking", "optical_internet",
        "heating_was_missing", "geo_missing", "dist_to_center_km"
    ]:
        if c in df.columns:
            numeric.append(c)

    low_card = []
    for c in ["state", "heating"]:
        if c in df.columns:
            low_card.append(c)

    high_card = []
    for c in ["street"]:
        if c in df.columns:
            high_card.append(c)

    return FeatureSpec(
        numeric=numeric,
        categorical_low_card=low_card,
        categorical_high_card=high_card
    )


def build_preprocess_pipeline(spec: FeatureSpec) -> ColumnTransformer:
    transformers = []

    if spec.numeric:
        num_pipe = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ])
        transformers.append(("num", num_pipe, spec.numeric))

    if spec.categorical_low_card:
        cat_pipe = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ])
        transformers.append(("cat_ohe", cat_pipe, spec.categorical_low_card))

    if spec.categorical_high_card:
        if TargetEncoder is not None:
            # IMPORTANT:
            # - No imputer before TargetEncoder (keeps DataFrame, avoids numpy array)
            # - No cols=... (ColumnTransformer already selects these columns)
            hi_pipe = Pipeline(steps=[
                ("te", TargetEncoder(
                    smoothing=5.0,
                    handle_missing="value",
                    handle_unknown="value"
                )),
            ])
            transformers.append(("cat_te", hi_pipe, spec.categorical_high_card))
        else:
            # Fallback: one-hot encode high-cardinality columns if category-encoders is not installed.
            hi_ohe = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
            ])
            transformers.append(("cat_high_ohe", hi_ohe, spec.categorical_high_card))

    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)


def split_xy(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    y = df[target_col].astype(float)
    X = df.drop(columns=[target_col])
    return X, y
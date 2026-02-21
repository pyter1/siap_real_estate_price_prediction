from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
from category_encoders.target_encoder import TargetEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


@dataclass(frozen=True)
class FeatureSpec:
    numeric: List[str]
    categorical_low_card: List[str]
    categorical_high_card: List[str]


def infer_feature_spec(df: pd.DataFrame) -> FeatureSpec:
    numeric = []
    for c in ["area_m2", "rooms", "floor", "lift", "parking", "optical_internet", "heating_was_missing", "geo_missing", "dist_to_center_km"]:
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

    return FeatureSpec(numeric=numeric, categorical_low_card=low_card, categorical_high_card=high_card)


def build_preprocess_pipeline(spec: FeatureSpec) -> ColumnTransformer:
    transformers = []

    if spec.numeric:
        transformers.append(("num", "passthrough", spec.numeric))

    if spec.categorical_low_card:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
        transformers.append(("cat_ohe", ohe, spec.categorical_low_card))

    if spec.categorical_high_card:
        te = TargetEncoder(cols=spec.categorical_high_card, smoothing=5.0)
        transformers.append(("cat_te", te, spec.categorical_high_card))

    return ColumnTransformer(transformers=transformers, remainder="drop", verbose_feature_names_out=False)


def split_xy(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    y = df[target_col].astype(float)
    X = df.drop(columns=[target_col])
    return X, y

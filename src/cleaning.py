from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd


_M2_RE = re.compile(r"([0-9]+(?:[.,][0-9]+)?)")


def parse_area_m2(value) -> float | None:
    if pd.isna(value):
        return None
    s = str(value).strip().lower()
    if s == "":
        return None
    m = _M2_RE.search(s)
    if not m:
        return None
    num = m.group(1).replace(",", ".")
    try:
        return float(num)
    except ValueError:
        return None



def parse_floor(value) -> float | None:
    """Parse floor values such as 'Prizemlje', 'Visoko prizemlje', 'Suteren', '3', '3.0'."""
    if pd.isna(value):
        return None
    s = str(value).strip().lower()
    if s == "":
        return None

    mapping = {
        "prizemlje": 0.0,
        "visoko prizemlje": 0.5,
        "suteren": -1.0,
    }
    if s in mapping:
        return mapping[s]

    # Common variants
    if "prizemlje" in s:
        return 0.0
    if "suteren" in s:
        return -1.0

    m = re.search(r"-?\d+(?:[.,]\d+)?", s)
    if not m:
        return None
    num = m.group(0).replace(",", ".")
    try:
        return float(num)
    except ValueError:
        return None

def parse_int01(value) -> int | None:
    if pd.isna(value):
        return None
    s = str(value).strip()
    if s == "":
        return None
    if s in {"0", "1"}:
        return int(s)
    try:
        v = int(float(s))
        if v in (0, 1):
            return v
        return None
    except ValueError:
        return None


def normalize_text(value) -> str | None:
    if pd.isna(value):
        return None
    s = str(value).strip()
    if s == "":
        return None
    s = re.sub(r"\s+", " ", s)
    return s


def normalize_heating(value) -> str | None:
    s = normalize_text(value)
    if s is None:
        return None
    s = s.lower()
    s = s.replace("грeјање", "grejanje")
    s = s.replace("grejane", "grejanje")
    return s


def normalize_location(value) -> str | None:
    s = normalize_text(value)
    if s is None:
        return None
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s


@dataclass
class CleaningReport:
    initial_rows: int
    dropped_missing_target: int
    dropped_missing_area: int
    dropped_invalid_target: int
    dropped_invalid_area: int
    dropped_outliers: int
    rental_like_filtered: int

    def to_dict(self) -> Dict[str, int]:
        return {
            "initial_rows": self.initial_rows,
            "dropped_missing_target": self.dropped_missing_target,
            "dropped_missing_area": self.dropped_missing_area,
            "dropped_invalid_target": self.dropped_invalid_target,
            "dropped_invalid_area": self.dropped_invalid_area,
            "dropped_outliers": self.dropped_outliers,
            "rental_like_filtered": self.rental_like_filtered,
        }


def filter_outliers_quantiles(
    df: pd.DataFrame,
    price_col: str,
    area_col: str,
    price_q: Tuple[float, float],
    area_q: Tuple[float, float],
) -> tuple[pd.DataFrame, int]:
    lo_p, hi_p = df[price_col].quantile(price_q[0]), df[price_col].quantile(price_q[1])
    lo_a, hi_a = df[area_col].quantile(area_q[0]), df[area_col].quantile(area_q[1])
    before = len(df)
    df2 = df[(df[price_col].between(lo_p, hi_p)) & (df[area_col].between(lo_a, hi_a))].copy()
    return df2, before - len(df2)


def filter_rental_like(df: pd.DataFrame, price_col: str, area_col: str) -> tuple[pd.DataFrame, int]:
    # Heuristic: rental listings tend to have low absolute price or very low price per m2.
    # Keep it conservative and rely on area presence.
    before = len(df)
    df2 = df.copy()

    # Drop rows with missing area, because many such rows represent inconsistent listings.
    df2 = df2[df2[area_col].notna()].copy()

    price_per_m2 = df2[price_col] / df2[area_col]
    # Thresholds tuned for sale listings in EUR.
    df2 = df2[(df2[price_col] >= 15000) & (price_per_m2 >= 300)].copy()

    return df2, before - len(df2)


def clean_dataset(
    df: pd.DataFrame,
    *,
    target_col: str,
    area_col_raw: str,
    rooms_col: str,
    state_col: str,
    lift_col: str,
    heating_col: str,
    optical_col: str,
    parking_col: str,
    floor_col: str,
    location_col: str,
    heating_default: str = "centralno grejanje",
    price_q=(0.005, 0.995),
    area_q=(0.005, 0.995),
) -> tuple[pd.DataFrame, CleaningReport]:
    initial_rows = len(df)

    out = df.copy()

    out[target_col] = pd.to_numeric(out[target_col], errors="coerce")
    dropped_missing_target = int(out[target_col].isna().sum())
    out = out[out[target_col].notna()].copy()

    dropped_invalid_target = int((out[target_col] <= 0).sum())
    out = out[out[target_col] > 0].copy()

    out["area_m2"] = out[area_col_raw].apply(parse_area_m2)
    dropped_missing_area = int(out["area_m2"].isna().sum())
    out = out[out["area_m2"].notna()].copy()

    dropped_invalid_area = int((out["area_m2"] <= 0).sum())
    out = out[out["area_m2"] > 0].copy()

    out["rooms"] = pd.to_numeric(out[rooms_col], errors="coerce")
    out["floor"] = out[floor_col].apply(parse_floor)

    out["lift"] = out[lift_col].apply(parse_int01)
    out["parking"] = out[parking_col].apply(parse_int01)
    out["optical_internet"] = out[optical_col].apply(parse_int01)

    out["state"] = out[state_col].apply(normalize_text).fillna("unknown")
    out["street_raw"] = out[location_col]
    out["street"] = out[location_col].apply(normalize_location)

    heating_missing = out[heating_col].isna() | (out[heating_col].astype(str).str.strip() == "")
    out["heating_was_missing"] = heating_missing.astype(int)
    out["heating"] = out[heating_col].apply(normalize_heating)
    out.loc[out["heating"].isna(), "heating"] = heating_default

    # Rental-like filter (optional but useful for your sample).
    out2, rental_filtered = filter_rental_like(out, price_col=target_col, area_col="area_m2")

    # Quantile outlier filter.
    out3, dropped_outliers = filter_outliers_quantiles(
        out2,
        price_col=target_col,
        area_col="area_m2",
        price_q=price_q,
        area_q=area_q,
    )

    report = CleaningReport(
        initial_rows=initial_rows,
        dropped_missing_target=dropped_missing_target,
        dropped_missing_area=dropped_missing_area,
        dropped_invalid_target=dropped_invalid_target,
        dropped_invalid_area=dropped_invalid_area,
        dropped_outliers=dropped_outliers,
        rental_like_filtered=rental_filtered,
    )

    return out3.reset_index(drop=True), report

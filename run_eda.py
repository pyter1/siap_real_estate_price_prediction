from __future__ import annotations

import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.cleaning import parse_area_m2, parse_floor
from src.config import SchemaConfig, get_paths
from src.io import load_csv


def _save_fig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/raw/listings.csv")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    paths = get_paths(project_root)
    schema = SchemaConfig()

    df = load_csv((project_root / args.input).resolve())

    # Basic parsing for EDA
    df = df.copy()
    df["area_m2"] = df[schema.area_col].apply(parse_area_m2)
    df["floor_num"] = df[schema.floor_col].apply(parse_floor)
    df[schema.target_col] = pd.to_numeric(df[schema.target_col], errors="coerce")

    summary = {
        "rows": int(len(df)),
        "missing": {c: float(df[c].isna().mean()) for c in df.columns},
        "price": {
            "min": float(df[schema.target_col].min(skipna=True)),
            "max": float(df[schema.target_col].max(skipna=True)),
            "median": float(df[schema.target_col].median(skipna=True)),
        },
        "area_m2": {
            "min": float(df["area_m2"].min(skipna=True)),
            "max": float(df["area_m2"].max(skipna=True)),
            "median": float(df["area_m2"].median(skipna=True)),
        },
        "unique_locations": int(df[schema.location_col].nunique(dropna=True)),
    }

    paths.reports.mkdir(parents=True, exist_ok=True)
    (paths.reports / "eda_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Plots
    plt.figure()
    df[schema.target_col].dropna().plot(kind="hist", bins=40)
    plt.xlabel("Price (EUR)")
    plt.title("Price distribution")
    _save_fig(paths.reports / "eda_price_hist.png")

    plt.figure()
    df["area_m2"].dropna().plot(kind="hist", bins=40)
    plt.xlabel("Area (m2)")
    plt.title("Area distribution")
    _save_fig(paths.reports / "eda_area_hist.png")

    # Scatter: area vs price
    tmp = df[[schema.target_col, "area_m2"]].dropna()
    tmp = tmp.sample(n=min(3000, len(tmp)), random_state=42)
    plt.figure()
    plt.scatter(tmp["area_m2"], tmp[schema.target_col], s=8, alpha=0.5)
    plt.xlabel("Area (m2)")
    plt.ylabel("Price (EUR)")
    plt.title("Area vs Price (sample)")
    _save_fig(paths.reports / "eda_area_vs_price.png")

    # Top categories
    for col, fname in [(schema.state_col, "eda_state_top.png"), (schema.heating_col, "eda_heating_top.png"), (schema.location_col, "eda_location_top.png")]:
        vc = df[col].astype(str).value_counts().head(20)
        plt.figure(figsize=(8, 6))
        plt.barh(range(len(vc))[::-1], vc.values[::-1])
        plt.yticks(range(len(vc))[::-1], vc.index[::-1])
        plt.title(f"Top 20: {col}")
        _save_fig(paths.reports / fname)

    print("EDA done.")
    print(f"Reports: {paths.reports}")


if __name__ == "__main__":
    main()

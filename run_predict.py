from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.cleaning import clean_dataset
from src.config import GeoConfig, SchemaConfig, get_paths
from src.geo import add_geo_features
from src.io import load_csv, require_columns, save_csv
from src.modeling import load_pipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/xgb_pipeline.joblib")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, default="data/processed/predictions.csv")
    parser.add_argument("--no-geo", action="store_true", help="Skip geocoding step")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    paths = get_paths(project_root)
    schema = SchemaConfig()
    geo_cfg = GeoConfig()

    model_path = (project_root / args.model).resolve()
    pipe = load_pipeline(str(model_path))

    input_path = (project_root / args.input).resolve()
    df_raw = load_csv(input_path)

    # Prediction input might include target column or not.
    # If it includes target, keep it for cleaning, then drop before prediction.
    required = [
        schema.area_col,
        schema.rooms_col,
        schema.state_col,
        schema.lift_col,
        schema.heating_col,
        schema.optical_col,
        schema.parking_col,
        schema.floor_col,
        schema.location_col,
    ]
    require_columns(df_raw, required)

    if schema.target_col in df_raw.columns:
        df_clean, _ = clean_dataset(
            df_raw,
            target_col=schema.target_col,
            area_col_raw=schema.area_col,
            rooms_col=schema.rooms_col,
            state_col=schema.state_col,
            lift_col=schema.lift_col,
            heating_col=schema.heating_col,
            optical_col=schema.optical_col,
            parking_col=schema.parking_col,
            floor_col=schema.floor_col,
            location_col=schema.location_col,
        )
        X = df_clean.drop(columns=[schema.target_col])
    else:
        # Add a dummy target to reuse cleaning code paths.
        tmp = df_raw.copy()
        tmp[schema.target_col] = 1.0
        df_clean, _ = clean_dataset(
            tmp,
            target_col=schema.target_col,
            area_col_raw=schema.area_col,
            rooms_col=schema.rooms_col,
            state_col=schema.state_col,
            lift_col=schema.lift_col,
            heating_col=schema.heating_col,
            optical_col=schema.optical_col,
            parking_col=schema.parking_col,
            floor_col=schema.floor_col,
            location_col=schema.location_col,
        )
        X = df_clean.drop(columns=[schema.target_col])

    if not args.no_geo:
        X = add_geo_features(
            X,
            location_col="street",
            cache_path=paths.geo_cache / "geocoded.csv",
            user_agent=geo_cfg.user_agent,
            city_hint=geo_cfg.city_hint,
            country_hint=geo_cfg.country_hint,
            delay_seconds=geo_cfg.request_delay_seconds,
            center_lat=geo_cfg.center_lat,
            center_lon=geo_cfg.center_lon,
        )

    preds = pipe.predict(X)
    out = X.copy()
    out["Predicted_Price_EUR"] = preds

    output_path = (project_root / args.output).resolve()
    save_csv(out, output_path)

    print(f"Saved predictions to: {output_path}")


if __name__ == "__main__":
    main()

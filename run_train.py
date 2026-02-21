from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.cleaning import clean_dataset
from src.config import GeoConfig, SchemaConfig, TrainConfig, get_paths
from src.geo import add_geo_features
from src.io import load_csv, require_columns, save_csv
from src.interpretability import get_model_importance, plot_importance, shap_summary_plot
from src.modeling import train_models, save_pipeline


def _plot_actual_vs_pred(y_true: np.ndarray, y_pred: np.ndarray, out_path: Path, title: str) -> None:
    plt.figure()
    plt.scatter(y_true, y_pred, s=8, alpha=0.6)
    mn = float(np.min([y_true.min(), y_pred.min()]))
    mx = float(np.max([y_true.max(), y_pred.max()]))
    plt.plot([mn, mx], [mn, mx])
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def _plot_residuals(y_true: np.ndarray, y_pred: np.ndarray, out_path: Path, title: str) -> None:
    res = y_true - y_pred
    plt.figure()
    plt.hist(res, bins=40)
    plt.xlabel("Residual (actual - predicted)")
    plt.ylabel("Count")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/raw/listings.csv")
    parser.add_argument("--no-geo", action="store_true", help="Skip geocoding step")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    paths = get_paths(project_root)
    schema = SchemaConfig()
    train_cfg = TrainConfig()
    geo_cfg = GeoConfig()

    input_path = (project_root / args.input).resolve()
    df_raw = load_csv(input_path)

    required = [
        schema.target_col,
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

    df_clean, cleaning_report = clean_dataset(
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
        price_q=(train_cfg.price_lower_quantile, train_cfg.price_upper_quantile),
        area_q=(train_cfg.area_lower_quantile, train_cfg.area_upper_quantile),
    )

    if not args.no_geo:
        df_clean = add_geo_features(
            df_clean,
            location_col="street",
            cache_path=paths.geo_cache / "geocoded.csv",
            user_agent=geo_cfg.user_agent,
            city_hint=geo_cfg.city_hint,
            country_hint=geo_cfg.country_hint,
            delay_seconds=geo_cfg.request_delay_seconds,
            center_lat=geo_cfg.center_lat,
            center_lon=geo_cfg.center_lon,
        )

    processed_path = paths.data_processed / "cleaned_with_features.csv"
    save_csv(df_clean, processed_path)

    artifacts, debug = train_models(
        df_clean,
        target_col=schema.target_col,
        random_seed=train_cfg.random_seed,
        test_size=train_cfg.test_size,
        cv_folds=train_cfg.cv_folds,
        n_iter_search=train_cfg.n_iter_search,
    )

    metrics_out = {
        "cleaning": cleaning_report.to_dict(),
        "debug": debug,
        "random_forest": artifacts.rf_metrics,
        "xgboost": artifacts.xgb_metrics,
    }
    (paths.reports).mkdir(parents=True, exist_ok=True)
    (paths.reports / "metrics.json").write_text(json.dumps(metrics_out, indent=2), encoding="utf-8")

    # Save models
    save_pipeline(artifacts.rf_pipeline, str(paths.models / "rf_pipeline.joblib"))
    save_pipeline(artifacts.xgb_pipeline, str(paths.models / "xgb_pipeline.joblib"))

    # Diagnostics plots based on test predictions computed via pipeline
    X = df_clean.drop(columns=[schema.target_col])
    y = df_clean[schema.target_col].astype(float).values

    # Use the XGBoost pipeline for plots
    y_pred = artifacts.xgb_pipeline.predict(X)

    _plot_actual_vs_pred(y, y_pred, paths.reports / "actual_vs_pred_xgb.png", "Actual vs Predicted (XGBoost)")
    _plot_residuals(y, y_pred, paths.reports / "residuals_xgb.png", "Residuals (XGBoost)")

    # Feature importance plots
    rf_imp = get_model_importance(artifacts.rf_pipeline, top_n=30)
    xgb_imp = get_model_importance(artifacts.xgb_pipeline, top_n=30)
    plot_importance(rf_imp, str(paths.reports / "importance_rf.png"), "Feature Importance (Random Forest)")
    plot_importance(xgb_imp, str(paths.reports / "importance_xgb.png"), "Feature Importance (XGBoost)")

    # SHAP on a sample
    sample = df_clean.drop(columns=[schema.target_col]).sample(n=min(500, len(df_clean)), random_state=train_cfg.random_seed)
    shap_summary_plot(artifacts.xgb_pipeline, sample, str(paths.reports / "shap_summary_xgb.png"), max_display=30)

    print("Training done.")
    print(f"Processed data: {processed_path}")
    print(f"Models: {paths.models}")
    print(f"Reports: {paths.reports}")


if __name__ == "__main__":
    main()

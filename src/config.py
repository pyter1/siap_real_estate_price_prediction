from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Paths:
    project_root: Path
    data_raw: Path
    data_processed: Path
    geo_cache: Path
    models: Path
    reports: Path


@dataclass(frozen=True)
class GeoConfig:
    user_agent: str = "real-estate-price-prediction"
    country_hint: str = "Srbija"
    city_hint: str = "Beograd"
    request_delay_seconds: float = 1.2
    center_lat: float = 44.8176
    center_lon: float = 20.4633


@dataclass(frozen=True)
class TrainConfig:
    random_seed: int = 42
    test_size: float = 0.2
    cv_folds: int = 5
    n_iter_search: int = 30
    price_lower_quantile: float = 0.005
    price_upper_quantile: float = 0.995
    area_lower_quantile: float = 0.005
    area_upper_quantile: float = 0.995


@dataclass(frozen=True)
class SchemaConfig:
    target_col: str = "Price_EUR"
    area_col: str = "Square_footage"
    rooms_col: str = "Number_of_rooms"
    state_col: str = "State"
    lift_col: str = "Lift"
    heating_col: str = "Heating"
    optical_col: str = "Optical_internet"
    parking_col: str = "Parking"
    floor_col: str = "Floor"
    location_col: str = "Street"


def get_paths(project_root: Path) -> Paths:
    return Paths(
        project_root=project_root,
        data_raw=project_root / "data" / "raw",
        data_processed=project_root / "data" / "processed",
        geo_cache=project_root / "data" / "geo_cache",
        models=project_root / "models",
        reports=project_root / "reports",
    )

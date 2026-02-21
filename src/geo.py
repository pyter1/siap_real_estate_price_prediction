from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from geopy.geocoders import Nominatim


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1 = np.radians(lat1)
    p2 = np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)

    a = np.sin(dlat / 2.0) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return float(r * c)


@dataclass
class GeoResult:
    lat: float | None
    lon: float | None
    status: str


class GeoCoder:
    def __init__(
        self,
        *,
        cache_path: Path,
        user_agent: str,
        city_hint: str,
        country_hint: str,
        delay_seconds: float,
    ):
        self.cache_path = cache_path
        self.city_hint = city_hint
        self.country_hint = country_hint
        self.delay_seconds = delay_seconds
        self.geolocator = Nominatim(user_agent=user_agent, timeout=10)

        self._cache: Dict[str, GeoResult] = {}
        self._load_cache()

    def _load_cache(self) -> None:
        if not self.cache_path.exists():
            return
        df = pd.read_csv(self.cache_path)
        for _, row in df.iterrows():
            key = str(row["key"])
            lat = None if pd.isna(row["lat"]) else float(row["lat"])
            lon = None if pd.isna(row["lon"]) else float(row["lon"])
            status = str(row["status"])
            self._cache[key] = GeoResult(lat=lat, lon=lon, status=status)

    def _save_cache(self) -> None:
        rows = []
        for key, gr in self._cache.items():
            rows.append({"key": key, "lat": gr.lat, "lon": gr.lon, "status": gr.status})
        out = pd.DataFrame(rows).sort_values("key")
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(self.cache_path, index=False)

    def _query_variants(self, location_text: str) -> list[str]:
        base = location_text.strip()
        variants = []
        if base:
            variants.append(f"{base}, {self.city_hint}, {self.country_hint}")
            variants.append(f"{base}, {self.city_hint}")
            variants.append(base)
        return variants

    def geocode_one(self, location_text: str) -> GeoResult:
        key = str(location_text).strip()
        if key == "" or key.lower() == "nan":
            return GeoResult(lat=None, lon=None, status="empty")

        if key in self._cache:
            return self._cache[key]

        result = GeoResult(lat=None, lon=None, status="failed")
        for q in self._query_variants(key):
            try:
                loc = self.geolocator.geocode(q)
            except Exception:
                loc = None
            time.sleep(self.delay_seconds)
            if loc is not None and loc.latitude is not None and loc.longitude is not None:
                result = GeoResult(lat=float(loc.latitude), lon=float(loc.longitude), status="ok")
                break

        self._cache[key] = result
        self._save_cache()
        return result


def add_geo_features(
    df: pd.DataFrame,
    *,
    location_col: str,
    cache_path: Path,
    user_agent: str,
    city_hint: str,
    country_hint: str,
    delay_seconds: float,
    center_lat: float,
    center_lon: float,
) -> pd.DataFrame:
    out = df.copy()
    gc = GeoCoder(
        cache_path=cache_path,
        user_agent=user_agent,
        city_hint=city_hint,
        country_hint=country_hint,
        delay_seconds=delay_seconds,
    )

    lats = []
    lons = []
    status = []

    for v in out[location_col].fillna("").astype(str).tolist():
        r = gc.geocode_one(v)
        lats.append(r.lat)
        lons.append(r.lon)
        status.append(r.status)

    out["lat"] = lats
    out["lon"] = lons
    out["geo_status"] = status
    out["geo_missing"] = out["lat"].isna().astype(int)

    dist = []
    for lat, lon in zip(out["lat"], out["lon"]):
        if pd.isna(lat) or pd.isna(lon):
            dist.append(np.nan)
        else:
            dist.append(haversine_km(float(lat), float(lon), center_lat, center_lon))
    out["dist_to_center_km"] = dist
    return out

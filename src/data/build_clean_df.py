from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import meteostat as ms

# project root (two levels above src/data/)
ROOT = Path(__file__).resolve().parents[2]

DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"

# ----------------------------
# Config
# ----------------------------
@dataclass(frozen=True)
class WeatherConfig:
    lat: float = 41.8781
    lon: float = -87.6298
    stations_limit: int = 4
    # Meteostat daily columns you may want to drop (optional)
    drop_cols: tuple[str, ...] = ("time", "cldc")


# ----------------------------
# Helpers
# ----------------------------
def _read_concat_csv(paths: Iterable[str | Path]) -> pd.DataFrame:
    dfs = []
    for p in paths:
        p = Path(p)
        dfs.append(pd.read_csv(p))
    return pd.concat(dfs, ignore_index=True)


def _coerce_trip_columns(trips: pd.DataFrame) -> pd.DataFrame:
    trips = trips.copy()

    # IDs
    trips["from_station_id"] = pd.to_numeric(trips["from_station_id"], errors="coerce")
    trips["to_station_id"] = pd.to_numeric(trips["to_station_id"], errors="coerce")

    # Times
    trips["start_time"] = pd.to_datetime(trips["start_time"], format="mixed", errors="coerce")
    trips["end_time"] = pd.to_datetime(trips["end_time"], format="mixed", errors="coerce")

    # Basic sanity: keep only rows with start_time
    trips = trips.dropna(subset=["start_time"]).copy()

    return trips


def _coerce_station_columns(stations: pd.DataFrame) -> pd.DataFrame:
    stations = stations.copy()
    stations["id"] = pd.to_numeric(stations["id"], errors="coerce")
    # Keep only what we need (rename later)
    return stations


def _merge_station_info(trips: pd.DataFrame, stations: pd.DataFrame) -> pd.DataFrame:
    # from station
    from_cols = (
        stations.rename(
            columns={
                "id": "from_station_id",
                "latitude": "from_lat",
                "longitude": "from_lon",
                "dpcapacity": "from_capacity",
            }
        )[["from_station_id", "from_lat", "from_lon", "from_capacity"]]
    )

    trips = trips.merge(from_cols, on="from_station_id", how="left")

    # to station
    to_cols = (
        stations.rename(
            columns={
                "id": "to_station_id",
                "latitude": "to_lat",
                "longitude": "to_lon",
                "dpcapacity": "to_capacity",
            }
        )[["to_station_id", "to_lat", "to_lon", "to_capacity"]]
    )

    trips = trips.merge(to_cols, on="to_station_id", how="left")
    return trips


def fetch_daily_weather(
    start_date,
    end_date,
    cfg: WeatherConfig = WeatherConfig(),
) -> pd.DataFrame:
    """
    Fetch daily weather between [start_date, end_date] inclusive and return a DataFrame
    with a 'date' column (python date) + Meteostat variables.
    """
    point = ms.Point(cfg.lat, cfg.lon)
    nearby = ms.stations.nearby(point, limit=cfg.stations_limit)

    ts = ms.daily(nearby, start_date, end_date)
    weather = ms.interpolate(ts, point).fetch().reset_index()

    # Meteostat returns a datetime-like in 'time'
    weather["date"] = pd.to_datetime(weather["time"]).dt.date

    # Drop unwanted columns if present
    drop_cols = [c for c in cfg.drop_cols if c in weather.columns]
    if drop_cols:
        weather = weather.drop(columns=drop_cols)

    return weather


def build_hourly_demand(trips: pd.DataFrame) -> pd.DataFrame:
    """
    Build station-hour grid with departures, arrivals, net_demand.
    """
    df = trips.copy()

    # Floor to hour
    df["start_hour"] = df["start_time"].dt.floor("h")
    df["end_hour"] = df["end_time"].dt.floor("h")

    departures = (
        df.groupby(["from_station_id", "start_hour"])
        .size()
        .reset_index(name="departures")
        .rename(columns={"from_station_id": "station_id", "start_hour": "hour"})
    )

    arrivals = (
        df.groupby(["to_station_id", "end_hour"])
        .size()
        .reset_index(name="arrivals")
        .rename(columns={"to_station_id": "station_id", "end_hour": "hour"})
    )

    # Full grid over stations x hours
    station_ids = pd.Index(
        pd.concat([df["from_station_id"], df["to_station_id"]], ignore_index=True)
        .dropna()
        .unique(),
        name="station_id",
    )

    min_hour = min(df["start_hour"].min(), df["end_hour"].min())
    max_hour = max(df["start_hour"].max(), df["end_hour"].max())

    hours = pd.date_range(min_hour, max_hour, freq="h", name="hour")

    calendar = pd.MultiIndex.from_product([station_ids, hours], names=["station_id", "hour"]).to_frame(index=False)

    demand = (
        calendar.merge(departures, on=["station_id", "hour"], how="left")
        .merge(arrivals, on=["station_id", "hour"], how="left")
    )

    demand["departures"] = demand["departures"].fillna(0).astype("int32")
    demand["arrivals"] = demand["arrivals"].fillna(0).astype("int32")
    demand["net_demand"] = (demand["arrivals"] - demand["departures"]).astype("int32")

    return demand


def add_weather_to_hourly_demand(
    demand: pd.DataFrame,
    weather_daily: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge daily weather onto hourly demand via demand['hour'].date.
    """
    out = demand.copy()
    out["date"] = out["hour"].dt.date
    out = out.merge(weather_daily, on="date", how="left")
    out = out.drop(columns=["date"])
    return out


# ----------------------------
# One main function
# ----------------------------
def build_divvy_datasets(
    trip_paths: Iterable[str | Path],
    stations_path: str | Path,
    *,
    save_clean_trips_to: Optional[str | Path] = None,
    save_demand_to: Optional[str | Path] = None,
    weather_cfg: WeatherConfig = WeatherConfig(),
) -> pd.DataFrame:
    """
    End-to-end:
    - read trips + stations
    - clean + merge station metadata
    - build hourly demand
    - fetch weather once and merge into demand
    - optionally save outputs

    Returns: hourly demand dataframe (with weather columns).
    """
    trips = _read_concat_csv(trip_paths)
    trips = _coerce_trip_columns(trips)

    stations = pd.read_csv(stations_path)
    stations = _coerce_station_columns(stations)

    trips = _merge_station_info(trips, stations)

    if save_clean_trips_to is not None:
        p = Path(save_clean_trips_to)
        p.parent.mkdir(parents=True, exist_ok=True)
        trips.to_csv(p, index=False)
        print("Saved clean trips:", p)

    demand = build_hourly_demand(trips)

    # Weather range based on hourly demand coverage
    start = demand["hour"].min().date()
    end = demand["hour"].max().date()
    weather = fetch_daily_weather(start, end, cfg=weather_cfg)

    demand = add_weather_to_hourly_demand(demand, weather)

    if save_demand_to is not None:
        p = Path(save_demand_to)
        p.parent.mkdir(parents=True, exist_ok=True)
        demand.to_csv(p, index=False)
        print("Saved demand dataset:", p)

    return demand


# ----------------------------
# CLI entrypoint
# ----------------------------
if __name__ == "__main__":
    demand = build_divvy_datasets(
        trip_paths=[
            DATA_RAW / "Divvy_Trips_2017_Q1.csv",
            DATA_RAW / "Divvy_Trips_2017_Q2.csv",
            DATA_RAW / "Divvy_Trips_2017_Q3.csv",
            DATA_RAW / "Divvy_Trips_2017_Q4.csv",
        ],
        stations_path=DATA_RAW / "Divvy_Stations_2017_Q3Q4 (2).csv",
        save_demand_to=DATA_PROCESSED / "divvy_hourly_demand_weather.csv",
    )

    print(demand.head())
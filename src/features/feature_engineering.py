from __future__ import annotations

import numpy as np
import pandas as pd


def add_time_and_lag_features(
    demand: pd.DataFrame,
    *,
    lags: tuple[int, ...] = (1, 24, 168),
    drop_na_lags: bool = True,
) -> pd.DataFrame:
    """
    Adds:
      - calendar features (hour/day-of-week/month/is_weekend)
      - cyclical sin/cos encodings for hour, day-of-week, month
      - per-station lags for departures and arrivals

    Expects columns:
      - station_id
      - hour (datetime64)
      - departures, arrivals

    Returns:
      - demand with added features (and optionally dropped NA lag rows)
    """
    df = demand.copy()

    if "hour" not in df.columns:
        raise ValueError("Expected column 'hour' (datetime).")
    if not np.issubdtype(df["hour"].dtype, np.datetime64):
        df["hour"] = pd.to_datetime(df["hour"], errors="coerce")

    # Basic time features
    df["hour_of_day"] = df["hour"].dt.hour.astype("int16")
    df["day_of_week"] = df["hour"].dt.dayofweek.astype("int8")  # Mon=0
    df["month"] = df["hour"].dt.month.astype("int8")
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype("int8")

    # Cyclical encodings
    df["hour_sin"] = np.sin(2 * np.pi * df["hour_of_day"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour_of_day"] / 24)

    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # Sort before lags (critical)
    df = df.sort_values(["station_id", "hour"]).reset_index(drop=True)

    # Lags per station
    for lag in lags:
        df[f"dep_lag_{lag}"] = df.groupby("station_id")["departures"].shift(lag)
        df[f"arr_lag_{lag}"] = df.groupby("station_id")["arrivals"].shift(lag)

    lag_cols = [f"dep_lag_{lag}" for lag in lags] + [f"arr_lag_{lag}" for lag in lags]

    if drop_na_lags:
        df = df.dropna(subset=lag_cols).copy()

    return df
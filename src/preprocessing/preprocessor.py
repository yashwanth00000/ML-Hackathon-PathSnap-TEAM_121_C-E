
"""
preprocessor.py
===============
Scikit-learn-compatible GPS pre-processing pipeline component.

GPSPreprocessor
---------------
* Drops GPS rows with NaN coordinates or out-of-range values.
* Optionally fills sub-second timestamp gaps (smooths to 1-second grid).
* Optionally applies a simple moving-average spatial smoother.
* Can be serialised/deserialised with joblib.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


# Coordinate validity bounds
LON_MIN, LON_MAX = -180.0, 180.0
LAT_MIN, LAT_MAX = -90.0, 90.0


class GPSPreprocessor(BaseEstimator, TransformerMixin):
    """Clean and smooth a GPS track DataFrame."""

    def __init__(
        self,
        smooth_window: int = 1,
        fill_gaps: bool = True,
        lon_range: tuple[float, float] = (LON_MIN, LON_MAX),
        lat_range: tuple[float, float] = (LAT_MIN, LAT_MAX),
    ) -> None:

        if not isinstance(smooth_window, int) or isinstance(smooth_window, bool) or smooth_window < 1:
            raise ValueError("smooth_window must be a positive integer")

        if len(lon_range) != 2 or lon_range[0] >= lon_range[1]:
            raise ValueError("lon_range must contain valid minimum and maximum values")

        if len(lat_range) != 2 or lat_range[0] >= lat_range[1]:
            raise ValueError("lat_range must contain valid minimum and maximum values")

        self.smooth_window = smooth_window
        self.fill_gaps = fill_gaps
        self.lon_range = lon_range
        self.lat_range = lat_range

    def fit(self, X: pd.DataFrame, y=None) -> "GPSPreprocessor":
        """No fitting required; returns self."""
        self.n_dropped_invalid_ = 0
        self.n_interpolated_ = 0
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """Clean and smooth a GPS track."""

        required_columns = {"lon", "lat", "timestamp"}
        missing_columns = required_columns.difference(X.columns)

        if missing_columns:
            raise ValueError(
                f"Input DataFrame is missing required columns: {sorted(missing_columns)}"
            )

        df = X.copy()
        before = len(df)

        # 1. Remove missing coordinates and timestamps
        df = df.dropna(subset=["lon", "lat", "timestamp"])

        # 2. Remove invalid geographic coordinates
        lon_ok = df["lon"].between(*self.lon_range)
        lat_ok = df["lat"].between(*self.lat_range)
        df = df[lon_ok & lat_ok]

        self.n_dropped_invalid_ = before - len(df)

        df = df.sort_values("timestamp").reset_index(drop=True)

        # 3. Fill small timestamp gaps
        if self.fill_gaps and len(df) > 1:
            df, n_interp = self._fill_timestamp_gaps(df)
            self.n_interpolated_ = n_interp
        else:
            self.n_interpolated_ = 0

        # 4. Apply spatial moving average
        if self.smooth_window > 1 and len(df) >= self.smooth_window:
            df = self._moving_average(df, self.smooth_window)

        return df.reset_index(drop=True)

    @staticmethod
    def _fill_timestamp_gaps(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
        """Interpolate missing rows for gaps up to 10 seconds."""

        MAX_GAP_FILL = 10

        rows = []
        n_added = 0

        for i in range(len(df) - 1):
            rows.append(df.iloc[i])

            gap = df["timestamp"].iloc[i + 1] - df["timestamp"].iloc[i]

            if 1 < gap <= MAX_GAP_FILL:
                n_steps = int(round(gap)) - 1

                for s in range(1, n_steps + 1):
                    alpha = s / round(gap)
                    new_row = df.iloc[i].copy()

                    new_row["timestamp"] = df["timestamp"].iloc[i] + s
                    new_row["lon"] = (
                        df["lon"].iloc[i] * (1 - alpha)
                        + df["lon"].iloc[i + 1] * alpha
                    )
                    new_row["lat"] = (
                        df["lat"].iloc[i] * (1 - alpha)
                        + df["lat"].iloc[i + 1] * alpha
                    )

                    rows.append(new_row)
                    n_added += 1

        rows.append(df.iloc[-1])

        return pd.DataFrame(rows).reset_index(drop=True), n_added

    @staticmethod
    def _moving_average(df: pd.DataFrame, window: int) -> pd.DataFrame:
        """Apply a centred moving average to longitude and latitude."""

        df = df.copy()
        half = window // 2

        df["lon"] = (
            df["lon"]
            .rolling(window, center=True, min_periods=max(1, half))
            .mean()
        )

        df["lat"] = (
            df["lat"]
            .rolling(window, center=True, min_periods=max(1, half))
            .mean()
        )

        return df

    def save(self, path: str | Path) -> None:
        """Save the preprocessor using joblib."""
        joblib.dump(self, path)
        print(f"[GPSPreprocessor] saved -> {path}")

    @classmethod
    def load(cls, path: str | Path) -> "GPSPreprocessor":
        """Load a previously saved preprocessor."""

        obj = joblib.load(path)

        if not isinstance(obj, cls):
            raise TypeError(f"Expected GPSPreprocessor, got {type(obj)}")

        print(f"[GPSPreprocessor] loaded <- {path}")
        return obj


if __name__ == "__main__":
    import io
    import textwrap
    import tempfile
    import os

    raw_csv = textwrap.dedent("""\
        lon,lat,timestamp
        -122.4194,37.7749,1609459200
        -122.4190,37.7752,1609459204
        NaN,37.7760,1609459205
        -999.0,37.7761,1609459206
        -122.4185,37.7755,1609459208
        -122.4180,37.7758,1609459209
    """)

    track = pd.read_csv(io.StringIO(raw_csv))

    pre = GPSPreprocessor(smooth_window=3, fill_gaps=True)
    clean = pre.fit_transform(track)

    print(f"Original rows: {len(track)}")
    print(f"Dropped invalid: {pre.n_dropped_invalid_}")
    print(f"Interpolated: {pre.n_interpolated_}")
    print(f"Output rows: {len(clean)}")

    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tf:
        tmp_path = tf.name

    try:
        pre.save(tmp_path)
        pre2 = GPSPreprocessor.load(tmp_path)
        clean2 = pre2.transform(track)

        assert len(clean) == len(clean2), "Round-trip mismatch!"
        print("Round-trip save/load: OK")
    finally:
        os.unlink(tmp_path)

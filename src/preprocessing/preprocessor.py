"""
preprocessor.py
===============
Scikit-learn–compatible GPS pre-processing pipeline component.

GPSPreprocessor
---------------
* Drops GPS rows with NaN coordinates or out-of-range values.
* Optionally fills sub-second timestamp gaps (smooths to 1-second grid).
* Optionally applies a simple moving-average spatial smoother.
* Can be serialised/deserialised with joblib.

Usage
-----
>>> from pathsnap.src.preprocessing.preprocessor import GPSPreprocessor
>>> pre = GPSPreprocessor(smooth_window=3)
>>> clean = pre.fit_transform(track_df)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


# ---------------------------------------------------------------------------
# Coordinate validity bounds
# ---------------------------------------------------------------------------

LON_MIN, LON_MAX = -180.0, 180.0
LAT_MIN, LAT_MAX = -90.0, 90.0


class GPSPreprocessor(BaseEstimator, TransformerMixin):
    """Clean and smooth a GPS track DataFrame.

    Parameters
    ----------
    smooth_window : int, optional
        Size of the moving-average window applied to ``lon`` and ``lat``.
        ``1`` (default) disables smoothing.
    fill_gaps : bool, optional
        If ``True`` (default), interpolate 1-second timestamp gaps in position.
    lon_range : tuple[float, float], optional
        Valid longitude bounds.  Rows outside are dropped.
    lat_range : tuple[float, float], optional
        Valid latitude bounds.  Rows outside are dropped.

    Attributes
    ----------
    n_dropped_invalid_ : int
        Number of rows removed due to NaN or out-of-range coordinates.
    n_interpolated_ : int
        Number of rows added by gap-filling (0 when ``fill_gaps=False``).
    """

    def __init__(
        self,
        smooth_window: int = 1,
        fill_gaps: bool = True,
        lon_range: tuple[float, float] = (LON_MIN, LON_MAX),
        lat_range: tuple[float, float] = (LAT_MIN, LAT_MAX),
    ) -> None:
        self.smooth_window = smooth_window
        self.fill_gaps = fill_gaps
        self.lon_range = lon_range
        self.lat_range = lat_range

    # ------------------------------------------------------------------
    # sklearn interface
    # ------------------------------------------------------------------

    def fit(self, X: pd.DataFrame, y=None) -> "GPSPreprocessor":
        """No fitting required; returns self.

        Parameters
        ----------
        X : pd.DataFrame
            GPS track with columns ``lon``, ``lat``, ``timestamp``.
        y : ignored

        Returns
        -------
        self
        """
        # Nothing to learn from data — kept for pipeline compatibility.
        self.n_dropped_invalid_ = 0
        self.n_interpolated_ = 0
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """Apply cleaning and smoothing to a GPS track.

        Parameters
        ----------
        X : pd.DataFrame
            GPS track with at least columns ``lon``, ``lat``, ``timestamp``.
        y : ignored

        Returns
        -------
        pd.DataFrame
            Cleaned (and optionally smoothed) track, reset index.
        """
        df = X.copy()
        before = len(df)

        # 1. Drop rows with NaN in core columns
        df = df.dropna(subset=["lon", "lat", "timestamp"])

        # 2. Drop out-of-range coordinates
        lon_ok = df["lon"].between(*self.lon_range)
        lat_ok = df["lat"].between(*self.lat_range)
        df = df[lon_ok & lat_ok]

        self.n_dropped_invalid_ = before - len(df)

        df = df.sort_values("timestamp").reset_index(drop=True)

        # 3. Gap-filling: insert interpolated rows for 1-second gaps
        if self.fill_gaps and len(df) > 1:
            df, n_interp = self._fill_timestamp_gaps(df)
            self.n_interpolated_ = n_interp
        else:
            self.n_interpolated_ = 0

        # 4. Moving-average spatial smoothing
        if self.smooth_window > 1 and len(df) >= self.smooth_window:
            df = self._moving_average(df, self.smooth_window)

        return df.reset_index(drop=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _fill_timestamp_gaps(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
        """Interpolate missing 1-second rows in a sorted GPS track.

        Only gaps up to 10 seconds are filled; larger gaps (likely signal
        loss) are left as-is to avoid fabricating long stretches of data.

        Parameters
        ----------
        df : pd.DataFrame
            Sorted GPS track.

        Returns
        -------
        (filled_df, n_rows_added) : tuple
        """
        MAX_GAP_FILL = 10  # seconds

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

        filled = pd.DataFrame(rows).reset_index(drop=True)
        return filled, n_added

    @staticmethod
    def _moving_average(df: pd.DataFrame, window: int) -> pd.DataFrame:
        """Apply a centred moving-average to lon and lat columns.

        Parameters
        ----------
        df : pd.DataFrame
            Sorted GPS track.
        window : int
            Moving-average window size (must be odd for symmetric centring).

        Returns
        -------
        pd.DataFrame
            Track with smoothed ``lon`` and ``lat`` columns.
        """
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

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Persist the preprocessor to disk using joblib.

        Parameters
        ----------
        path : str or Path
            Destination file (e.g. ``preprocessor.joblib``).
        """
        joblib.dump(self, path)
        print(f"[GPSPreprocessor] saved → {path}")

    @classmethod
    def load(cls, path: str | Path) -> "GPSPreprocessor":
        """Load a previously saved preprocessor from disk.

        Parameters
        ----------
        path : str or Path
            Path to the joblib file.

        Returns
        -------
        GPSPreprocessor
        """
        obj = joblib.load(path)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected GPSPreprocessor, got {type(obj)}")
        print(f"[GPSPreprocessor] loaded ← {path}")
        return obj


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import textwrap, io

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

    print(f"Original rows : {len(track)}")
    print(f"Dropped invalid: {pre.n_dropped_invalid_}")
    print(f"Interpolated   : {pre.n_interpolated_}")
    print(f"Output rows    : {len(clean)}")
    print(clean.to_string(index=False))

    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tf:
        tmp_path = tf.name

    pre.save(tmp_path)
    pre2 = GPSPreprocessor.load(tmp_path)
    clean2 = pre2.transform(track)
    assert len(clean) == len(clean2), "Round-trip mismatch!"
    print("\nRound-trip save/load: OK")
    os.unlink(tmp_path)

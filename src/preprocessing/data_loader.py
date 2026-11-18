"""
data_loader.py
==============
Utilities for loading PathSnap GPS map-matching dataset files.

File formats (all whitespace-separated, no headers):
  .track  : lon  lat  timestamp
  .nodes  : lon  lat          (row index = node_id)
  .arcs   : from_node_id  to_node_id  (row index = arc_id)
  .route  : one arc_id per line (ground-truth sequence)
"""

from __future__ import annotations

import os
import glob
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Low-level file loaders
# ---------------------------------------------------------------------------

def load_track(path: str | Path) -> pd.DataFrame:
    """Load a .track file into a DataFrame.

    Parameters
    ----------
    path : str or Path
        Path to the .track file.

    Returns
    -------
    pd.DataFrame
        Columns: ``lon`` (float), ``lat`` (float), ``timestamp`` (float).
        Rows are in original order (chronological).
    """
    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=["lon", "lat", "timestamp"],
        dtype={"lon": float, "lat": float, "timestamp": float},
    )
    return df.reset_index(drop=True)


def load_nodes(path: str | Path) -> pd.DataFrame:
    """Load a .nodes file into a DataFrame.

    Parameters
    ----------
    path : str or Path
        Path to the .nodes file.

    Returns
    -------
    pd.DataFrame
        Columns: ``lon``, ``lat``.  Index is ``node_id`` (0-based integer).
    """
    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=["lon", "lat"],
        dtype=float,
    )
    df.index.name = "node_id"
    return df


def load_arcs(path: str | Path) -> pd.DataFrame:
    """Load a .arcs file into a DataFrame.

    Parameters
    ----------
    path : str or Path
        Path to the .arcs file.

    Returns
    -------
    pd.DataFrame
        Columns: ``from_node_id`` (int), ``to_node_id`` (int).
        Index is ``arc_id`` (0-based integer).
    """
    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=["from_node_id", "to_node_id"],
        dtype=int,
    )
    df.index.name = "arc_id"
    return df


def load_route(path: str | Path) -> List[int]:
    """Load a .route file into a list of arc IDs.

    Parameters
    ----------
    path : str or Path
        Path to the .route file.

    Returns
    -------
    list of int
        Ordered sequence of arc IDs representing the ground-truth route.
    """
    with open(path, "r") as fh:
        lines = fh.read().strip().splitlines()
    return [int(line.strip()) for line in lines if line.strip()]


# ---------------------------------------------------------------------------
# Trip-level loader
# ---------------------------------------------------------------------------

def load_trip(trip_dir: str | Path) -> Dict:
    """Load all files for a single trip from a directory.

    The directory is expected to contain exactly one file of each type:
    ``*.track``, ``*.nodes``, ``*.arcs``, and optionally ``*.route``.

    Parameters
    ----------
    trip_dir : str or Path
        Directory containing the trip files.

    Returns
    -------
    dict with keys:
        ``trip_id``  : str  — stem of the directory name
        ``track``    : pd.DataFrame  — GPS observations
        ``nodes``    : pd.DataFrame  — road-network nodes
        ``arcs``     : pd.DataFrame  — road-network arcs
        ``route``    : list[int] or None  — ground-truth arc sequence
        ``trip_dir`` : Path  — absolute path to the directory
    """
    trip_dir = Path(trip_dir)

    def _find_one(pattern: str) -> Optional[Path]:
        matches = list(trip_dir.glob(pattern))
        if not matches:
            return None
        if len(matches) > 1:
            # Prefer the one whose stem matches the directory name
            stem = trip_dir.name
            for m in matches:
                if m.stem == stem:
                    return m
        return matches[0]

    track_path = _find_one("*.track")
    nodes_path = _find_one("*.nodes")
    arcs_path = _find_one("*.arcs")
    route_path = _find_one("*.route")

    if track_path is None:
        raise FileNotFoundError(f"No .track file found in {trip_dir}")
    if nodes_path is None:
        raise FileNotFoundError(f"No .nodes file found in {trip_dir}")
    if arcs_path is None:
        raise FileNotFoundError(f"No .arcs file found in {trip_dir}")

    trip = {
        "trip_id": trip_dir.name,
        "track": load_track(track_path),
        "nodes": load_nodes(nodes_path),
        "arcs": load_arcs(arcs_path),
        "route": load_route(route_path) if route_path is not None else None,
        "trip_dir": trip_dir.resolve(),
    }
    return trip


# ---------------------------------------------------------------------------
# Partition-level loader
# ---------------------------------------------------------------------------

def load_partition(partition_dir: str | Path) -> List[Dict]:
    """Load all trips from a partition directory (train / test).

    Each sub-directory (or any directory containing a .track file) is treated
    as one trip.  Trips with no .track file are silently skipped.

    Parameters
    ----------
    partition_dir : str or Path
        Top-level directory for a data partition.

    Returns
    -------
    list of trip dicts (see :func:`load_trip` for the dict schema).
    """
    partition_dir = Path(partition_dir)
    trips: List[Dict] = []

    # Support two layouts:
    #   1. partition_dir/<trip_id>/<trip_id>.track  (one sub-dir per trip)
    #   2. partition_dir/<trip_id>.track            (flat layout)
    subdirs = sorted([p for p in partition_dir.iterdir() if p.is_dir()])

    if subdirs:
        for sub in subdirs:
            if list(sub.glob("*.track")):
                try:
                    trips.append(load_trip(sub))
                except Exception as exc:  # noqa: BLE001
                    print(f"[load_partition] Skipping {sub}: {exc}")
    else:
        # Flat layout: group by stem
        stems = {p.stem for p in partition_dir.glob("*.track")}
        for stem in sorted(stems):
            try:
                trips.append(load_trip(partition_dir))
            except Exception as exc:
                print(f"[load_partition] Skipping {stem}: {exc}")
            break  # flat layout means the whole dir is one trip

    return trips


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    base = Path(__file__).resolve().parents[3]  # repo root
    train_dir = base / "data" / "train"

    if not train_dir.exists():
        print(f"[demo] Training directory not found at {train_dir}.")
        print("[demo] Create synthetic data for a quick smoke-test …")

        import tempfile, textwrap

        with tempfile.TemporaryDirectory() as tmp:
            td = Path(tmp) / "trip_001"
            td.mkdir()
            (td / "trip_001.track").write_text(
                textwrap.dedent("""\
                    -122.4194 37.7749 1609459200
                    -122.4190 37.7752 1609459201
                    -122.4185 37.7755 1609459202
                """)
            )
            (td / "trip_001.nodes").write_text(
                textwrap.dedent("""\
                    -122.4194 37.7749
                    -122.4185 37.7755
                    -122.4175 37.7760
                """)
            )
            (td / "trip_001.arcs").write_text(
                textwrap.dedent("""\
                    0 1
                    1 2
                """)
            )
            (td / "trip_001.route").write_text("0\n1\n")

            trip = load_trip(td)
            print("trip_id  :", trip["trip_id"])
            print("track\n",   trip["track"])
            print("nodes\n",   trip["nodes"])
            print("arcs\n",    trip["arcs"])
            print("route     :", trip["route"])
    else:
        trips = load_partition(train_dir)
        print(f"Loaded {len(trips)} trips from {train_dir}")
        t0 = trips[0]
        print(f"  trip_id={t0['trip_id']}  "
              f"GPS obs={len(t0['track'])}  "
              f"nodes={len(t0['nodes'])}  "
              f"arcs={len(t0['arcs'])}  "
              f"route_len={len(t0['route']) if t0['route'] else 'N/A'}")

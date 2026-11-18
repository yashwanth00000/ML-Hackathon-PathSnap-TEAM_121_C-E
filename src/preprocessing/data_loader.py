
"""
data_loader.py
==============
Utilities for loading PathSnap GPS map-matching dataset files.

File formats (all whitespace-separated, no headers):
  .track  : lon  lat  timestamp
  .nodes  : lon  lat
  .arcs   : from_node_id  to_node_id
  .route  : one arc_id per line
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Low-level file loaders
# ---------------------------------------------------------------------------

def _validate_file(path: str | Path, extension: str) -> Path:
    """Check that a dataset file exists and has the expected extension."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    if not path.is_file():
        raise ValueError(f"Expected a file, got: {path}")

    if path.suffix != extension:
        raise ValueError(
            f"Expected a {extension} file, got: {path.name}"
        )

    return path


def load_track(path: str | Path) -> pd.DataFrame:
    """Load GPS observations from a .track file."""
    path = _validate_file(path, ".track")

    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=["lon", "lat", "timestamp"],
        dtype={
            "lon": float,
            "lat": float,
            "timestamp": float,
        },
    )

    return df.reset_index(drop=True)


def load_nodes(path: str | Path) -> pd.DataFrame:
    """Load road-network nodes from a .nodes file."""
    path = _validate_file(path, ".nodes")

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
    """Load road-network arcs from an .arcs file."""
    path = _validate_file(path, ".arcs")

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
    """Load the ground-truth sequence of arc IDs from a .route file."""
    path = _validate_file(path, ".route")

    with path.open("r", encoding="utf-8") as file:
        return [
            int(line.strip())
            for line in file
            if line.strip()
        ]


# ---------------------------------------------------------------------------
# Trip-level loader
# ---------------------------------------------------------------------------

def load_trip(
    trip_dir: str | Path,
    trip_id: Optional[str] = None,
) -> Dict:
    """
    Load all dataset files belonging to one trip.

    Files are matched using the trip ID when supplied. If there is only
    one file of a given type, it can be used even if its name differs.

    The .route file is optional.
    """
    trip_dir = Path(trip_dir)

    if not trip_dir.exists():
        raise FileNotFoundError(
            f"Trip directory does not exist: {trip_dir}"
        )

    if not trip_dir.is_dir():
        raise NotADirectoryError(
            f"Expected a trip directory: {trip_dir}"
        )

    selected_id = trip_id or trip_dir.name

    def find_file(extension: str, required: bool = True) -> Optional[Path]:
        matches = sorted(trip_dir.glob(f"*{extension}"))

        matching_file = next(
            (file for file in matches if file.stem == selected_id),
            None,
        )

        if matching_file is not None:
            return matching_file

        if len(matches) == 1:
            return matches[0]

        if not matches:
            if required:
                raise FileNotFoundError(
                    f"No {extension} file found in {trip_dir}"
                )
            return None

        raise ValueError(
            f"Multiple {extension} files found in {trip_dir}; "
            f"none matches trip ID '{selected_id}'"
        )

    track_path = find_file(".track")
    nodes_path = find_file(".nodes")
    arcs_path = find_file(".arcs")
    route_path = find_file(".route", required=False)

    return {
        "trip_id": selected_id,
        "track": load_track(track_path),
        "nodes": load_nodes(nodes_path),
        "arcs": load_arcs(arcs_path),
        "route": load_route(route_path) if route_path else None,
        "trip_dir": trip_dir.resolve(),
    }


# ---------------------------------------------------------------------------
# Partition-level loader
# ---------------------------------------------------------------------------

def load_partition(partition_dir: str | Path) -> List[Dict]:
    """
    Load all trips from a train/test partition.

    Supports both layouts:

    1. One directory per trip:
       partition/trip_001/trip_001.track

    2. All trip files in one directory:
       partition/trip_001.track
       partition/trip_002.track
    """
    partition_dir = Path(partition_dir)

    if not partition_dir.exists():
        raise FileNotFoundError(
            f"Partition directory does not exist: {partition_dir}"
        )

    if not partition_dir.is_dir():
        raise NotADirectoryError(
            f"Expected a partition directory: {partition_dir}"
        )

    trips: List[Dict] = []

    # Load trips stored in individual subdirectories.
    subdirs = sorted(
        path for path in partition_dir.iterdir()
        if path.is_dir()
    )

    for subdir in subdirs:
        if not any(subdir.glob("*.track")):
            continue

        try:
            trips.append(load_trip(subdir))
        except (OSError, ValueError, pd.errors.ParserError) as exc:
            print(f"[load_partition] Skipping {subdir}: {exc}")

    # Also load tracks stored directly inside the partition directory.
    flat_track_files = sorted(partition_dir.glob("*.track"))

    for track_file in flat_track_files:
        try:
            trips.append(
                load_trip(partition_dir, trip_id=track_file.stem)
            )
        except (OSError, ValueError, pd.errors.ParserError) as exc:
            print(
                f"[load_partition] Skipping "
                f"{track_file.stem}: {exc}"
            )

    return trips


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile
    import textwrap

    with tempfile.TemporaryDirectory() as tmp:
        partition = Path(tmp) / "train"
        partition.mkdir()

        # Create two trips in the flat directory layout.
        for trip_number in range(1, 3):
            trip_id = f"trip_00{trip_number}"

            (partition / f"{trip_id}.track").write_text(
                textwrap.dedent("""\
                    -122.4194 37.7749 1609459200
                    -122.4190 37.7752 1609459201
                    -122.4185 37.7755 1609459202
                """),
                encoding="utf-8",
            )

            (partition / f"{trip_id}.nodes").write_text(
                textwrap.dedent("""\
                    -122.4194 37.7749
                    -122.4185 37.7755
                    -122.4175 37.7760
                """),
                encoding="utf-8",
            )

            (partition / f"{trip_id}.arcs").write_text(
                "0 1\n1 2\n",
                encoding="utf-8",
            )

            (partition / f"{trip_id}.route").write_text(
                "0\n1\n",
                encoding="utf-8",
            )

        loaded_trips = load_partition(partition)

        print(f"Loaded trips: {len(loaded_trips)}")

        for trip in loaded_trips:
            print(
                f"{trip['trip_id']}: "
                f"GPS observations={len(trip['track'])}, "
                f"nodes={len(trip['nodes'])}, "
                f"arcs={len(trip['arcs'])}, "
                f"route length={len(trip['route'])}"
            )

        assert len(loaded_trips) == 2
        print("Flat-layout loading test: PASSED")

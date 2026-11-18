"""
feature_engineering.py
=======================
Feature extraction for the PathSnap GPS map-matching problem.

Candidate-arc generation
------------------------
For each GPS observation the top-k nearest arcs are selected as candidates.
Features are computed for every (GPS obs, candidate arc) pair and assembled
into a flat feature matrix suitable for scikit-learn models.

Feature columns (13 total)
--------------------------
  0  perp_dist_m          Perpendicular distance from GPS point to arc (metres)
  1  proj_ratio           Along-arc projection ratio in [0, 1] (clamped)
  2  proj_dist_m          Distance from arc start to foot-of-perpendicular (m)
  3  arc_length_m         Total arc length (metres)
  4  bearing_diff_deg     Absolute bearing difference between GPS heading and
                          arc direction (degrees, in [0, 90])
  5  gps_speed_mps        Estimated GPS speed (m/s), 0 if unavailable
  6  time_since_last_s    Seconds since previous GPS fix (accuracy proxy)
  7  node_start_dist_m    Haversine distance from GPS point to arc start node
  8  node_end_dist_m      Haversine distance from GPS point to arc end node
  9  candidate_rank       Rank of this candidate (0 = nearest arc)
  10 lon                  GPS longitude (raw)
  11 lat                  GPS latitude (raw)
  12 timestamp            GPS timestamp (raw)
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Geographic helpers
# ---------------------------------------------------------------------------

EARTH_RADIUS_M = 6_371_000.0  # mean Earth radius in metres


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute the great-circle distance between two geographic points.

    Parameters
    ----------
    lat1, lon1 : float
        Latitude and longitude of point 1 (degrees).
    lat2, lon2 : float
        Latitude and longitude of point 2 (degrees).

    Returns
    -------
    float
        Distance in metres.
    """
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(a))


def compute_heading(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute the forward bearing from point 1 to point 2.

    Parameters
    ----------
    lat1, lon1 : float
        Start latitude/longitude (degrees).
    lat2, lon2 : float
        End latitude/longitude (degrees).

    Returns
    -------
    float
        Bearing in degrees [0, 360), measured clockwise from north.
    """
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dlambda = math.radians(lon2 - lon1)

    x = math.sin(dlambda) * math.cos(phi2)
    y = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlambda)
    bearing = math.degrees(math.atan2(x, y))
    return bearing % 360.0


def point_to_segment_distance(
    px: float, py: float,
    ax: float, ay: float,
    bx: float, by: float,
) -> Tuple[float, float]:
    """Compute the distance from point P to line-segment AB in a flat (x,y) space.

    For geographic use, convert lon/lat to local metres first (see
    :func:`_to_local_xy`).

    Parameters
    ----------
    px, py : float
        Query point coordinates.
    ax, ay : float
        Segment start coordinates.
    bx, by : float
        Segment end coordinates.

    Returns
    -------
    (distance, proj_ratio) : tuple[float, float]
        ``distance``   — perpendicular distance from P to the nearest point on AB.
        ``proj_ratio`` — scalar t ∈ [0, 1] such that the foot of perpendicular
                         is at ``A + t*(B-A)``; 0 means foot = A, 1 means foot = B.
    """
    dx = bx - ax
    dy = by - ay
    seg_len_sq = dx * dx + dy * dy

    if seg_len_sq == 0.0:
        # Degenerate segment (A == B)
        return math.hypot(px - ax, py - ay), 0.0

    t = ((px - ax) * dx + (py - ay) * dy) / seg_len_sq
    t_clamped = max(0.0, min(1.0, t))

    foot_x = ax + t_clamped * dx
    foot_y = ay + t_clamped * dy

    dist = math.hypot(px - foot_x, py - foot_y)
    return dist, t_clamped


# ---------------------------------------------------------------------------
# Internal coordinate projection
# ---------------------------------------------------------------------------

def _to_local_xy(
    lon: float,
    lat: float,
    origin_lon: float,
    origin_lat: float,
) -> Tuple[float, float]:
    """Project (lon, lat) to local East-North metres around an origin.

    Uses an equirectangular approximation, accurate for small areas (<100 km).

    Parameters
    ----------
    lon, lat : float
        Target point (degrees).
    origin_lon, origin_lat : float
        Origin point (degrees).

    Returns
    -------
    (x, y) : tuple[float, float]
        East (x) and North (y) offsets in metres.
    """
    lat_mid = math.radians((lat + origin_lat) / 2.0)
    x = math.radians(lon - origin_lon) * EARTH_RADIUS_M * math.cos(lat_mid)
    y = math.radians(lat - origin_lat) * EARTH_RADIUS_M
    return x, y


# ---------------------------------------------------------------------------
# Arc-to-GPS features
# ---------------------------------------------------------------------------

FEATURE_COLUMNS = [
    "perp_dist_m",
    "proj_ratio",
    "proj_dist_m",
    "arc_length_m",
    "bearing_diff_deg",
    "gps_speed_mps",
    "time_since_last_s",
    "node_start_dist_m",
    "node_end_dist_m",
    "candidate_rank",
    "lon",
    "lat",
    "timestamp",
]


def _arc_endpoints(
    arc_row: pd.Series, nodes_df: pd.DataFrame
) -> Tuple[float, float, float, float]:
    """Return (lon_a, lat_a, lon_b, lat_b) for an arc.

    Parameters
    ----------
    arc_row : pd.Series
        A single row from an arcs DataFrame.
    nodes_df : pd.DataFrame
        Nodes table (lon, lat indexed by node_id).

    Returns
    -------
    tuple[float, float, float, float]
    """
    a = nodes_df.loc[arc_row["from_node_id"]]
    b = nodes_df.loc[arc_row["to_node_id"]]
    return float(a["lon"]), float(a["lat"]), float(b["lon"]), float(b["lat"])


def _features_for_arc(
    gps_lon: float,
    gps_lat: float,
    arc_row: pd.Series,
    nodes_df: pd.DataFrame,
    gps_speed: float,
    gps_heading: float,
    time_since_last: float,
    rank: int,
    gps_timestamp: float,
) -> List[float]:
    """Compute the feature vector for one (GPS obs, candidate arc) pair.

    Parameters
    ----------
    gps_lon, gps_lat : float
        GPS point coordinates.
    arc_row : pd.Series
        Arc row from arcs DataFrame.
    nodes_df : pd.DataFrame
        Nodes table.
    gps_speed : float
        Estimated GPS speed (m/s).
    gps_heading : float
        Estimated GPS heading (degrees, or NaN if unknown).
    time_since_last : float
        Seconds elapsed since the previous GPS fix.
    rank : int
        Candidate rank (0 = closest).
    gps_timestamp : float
        Raw GPS timestamp.

    Returns
    -------
    list[float]
        Feature values in ``FEATURE_COLUMNS`` order.
    """
    lon_a, lat_a, lon_b, lat_b = _arc_endpoints(arc_row, nodes_df)

    # --- project everything to local metres around the GPS point ---
    origin_lon, origin_lat = gps_lon, gps_lat
    px, py = 0.0, 0.0  # GPS point is origin
    ax, ay = _to_local_xy(lon_a, lat_a, origin_lon, origin_lat)
    bx, by = _to_local_xy(lon_b, lat_b, origin_lon, origin_lat)

    perp_dist_m, proj_ratio = point_to_segment_distance(px, py, ax, ay, bx, by)
    arc_length_m = math.hypot(bx - ax, by - ay)
    proj_dist_m = proj_ratio * arc_length_m

    # --- bearing difference ---
    arc_bearing = compute_heading(lat_a, lon_a, lat_b, lon_b)
    if not math.isnan(gps_heading):
        raw_diff = abs(gps_heading - arc_bearing) % 360.0
        bearing_diff_deg = min(raw_diff, 360.0 - raw_diff)
        # Collapse 180°–opposite arcs: a road is traversable in both directions
        bearing_diff_deg = min(bearing_diff_deg, 180.0 - bearing_diff_deg)
        bearing_diff_deg = abs(bearing_diff_deg)
    else:
        bearing_diff_deg = 45.0  # neutral fallback

    # --- node distances (haversine) ---
    node_start_dist_m = haversine(gps_lat, gps_lon, lat_a, lon_a)
    node_end_dist_m = haversine(gps_lat, gps_lon, lat_b, lon_b)

    return [
        perp_dist_m,
        proj_ratio,
        proj_dist_m,
        arc_length_m,
        bearing_diff_deg,
        gps_speed,
        time_since_last,
        node_start_dist_m,
        node_end_dist_m,
        float(rank),
        gps_lon,
        gps_lat,
        gps_timestamp,
    ]


def generate_candidates(
    gps_point: pd.Series,
    nodes_df: pd.DataFrame,
    arcs_df: pd.DataFrame,
    k: int = 5,
    gps_speed: float = 0.0,
    gps_heading: float = float("nan"),
    time_since_last: float = 0.0,
) -> pd.DataFrame:
    """Return the top-k candidate arcs for a single GPS observation.

    Arcs are ranked by perpendicular distance from the GPS point.

    Parameters
    ----------
    gps_point : pd.Series
        A row from a track DataFrame with at least ``lon``, ``lat``,
        and ``timestamp`` fields.
    nodes_df : pd.DataFrame
        Nodes table (indexed by node_id).
    arcs_df : pd.DataFrame
        Arcs table (indexed by arc_id).
    k : int, optional
        Number of candidate arcs to return (default 5).
    gps_speed : float, optional
        Estimated GPS speed in m/s (default 0).
    gps_heading : float, optional
        Estimated GPS heading in degrees (NaN → unknown).
    time_since_last : float, optional
        Seconds since the previous GPS fix.

    Returns
    -------
    pd.DataFrame
        One row per candidate arc, columns = ``FEATURE_COLUMNS`` + ``arc_id``.
        Sorted by ``perp_dist_m`` ascending.
    """
    gps_lon = float(gps_point["lon"])
    gps_lat = float(gps_point["lat"])
    gps_ts = float(gps_point["timestamp"])

    rows = []
    for rank, (arc_id, arc_row) in enumerate(arcs_df.iterrows()):
        feats = _features_for_arc(
            gps_lon=gps_lon,
            gps_lat=gps_lat,
            arc_row=arc_row,
            nodes_df=nodes_df,
            gps_speed=gps_speed,
            gps_heading=gps_heading,
            time_since_last=time_since_last,
            rank=0,  # will update after sorting
            gps_timestamp=gps_ts,
        )
        rows.append([arc_id] + feats)

    cols = ["arc_id"] + FEATURE_COLUMNS
    df = pd.DataFrame(rows, columns=cols)
    df = df.nsmallest(k, "perp_dist_m").reset_index(drop=True)
    df["candidate_rank"] = df.index.astype(float)
    return df


# ---------------------------------------------------------------------------
# Trip-level feature engineering
# ---------------------------------------------------------------------------

def engineer_features(
    trip_dict: Dict,
    k: int = 5,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Build the (X, y) feature matrix and label vector for one trip.

    For each GPS observation in the track, up to ``k`` candidate arcs are
    generated.  Each (observation, candidate) pair becomes one row in ``X``.

    Parameters
    ----------
    trip_dict : dict
        A trip dictionary as returned by ``data_loader.load_trip``.
        Required keys: ``track``, ``nodes``, ``arcs``.
        Optional key:  ``route`` (None for evaluator trips).
    k : int, optional
        Number of candidate arcs per GPS observation (default 5).

    Returns
    -------
    X : np.ndarray, shape (n_obs * k, len(FEATURE_COLUMNS))
        Feature matrix.  ``n_obs`` is the number of GPS observations after
        any pre-processing; each is paired with up to ``k`` candidate arcs.
    y : np.ndarray of int, shape (n_obs * k,) or None
        Binary labels: 1 if the candidate arc matches the ground-truth arc
        for that observation, 0 otherwise.  ``None`` when no route is
        available (evaluator trips).

    Notes
    -----
    Ground-truth matching uses the following heuristic: the ``i``-th GPS
    observation is paired with ``route[i]`` when the route length equals the
    number of GPS observations.  When lengths differ, the nearest arc from the
    route set is used as the ground-truth for each observation.
    """
    track: pd.DataFrame = trip_dict["track"]
    nodes_df: pd.DataFrame = trip_dict["nodes"]
    arcs_df: pd.DataFrame = trip_dict["arcs"]
    route: Optional[List[int]] = trip_dict.get("route")

    has_labels = route is not None
    route_set = set(route) if has_labels else set()

    X_rows: List[List[float]] = []
    y_vals: List[int] = []

    # Pre-compute per-observation speed and heading
    timestamps = track["timestamp"].values
    lons = track["lon"].values
    lats = track["lat"].values

    speeds = np.zeros(len(track))
    headings = np.full(len(track), float("nan"))
    times_since = np.zeros(len(track))

    for i in range(1, len(track)):
        dt = float(timestamps[i] - timestamps[i - 1])
        if dt > 0:
            dist = haversine(lats[i - 1], lons[i - 1], lats[i], lons[i])
            speeds[i] = dist / dt
            headings[i] = compute_heading(lats[i - 1], lons[i - 1], lats[i], lons[i])
            times_since[i] = dt
        else:
            speeds[i] = speeds[i - 1]
            headings[i] = headings[i - 1]
            times_since[i] = 0.0

    # Determine ground-truth arc per observation
    # Strategy: index-based if route length == track length, else nearest-in-route
    if has_labels and len(route) == len(track):
        gt_per_obs: Optional[List[int]] = route
    elif has_labels:
        # Fallback: assign the route arc whose midpoint is closest to each obs
        route_arcs = arcs_df.loc[arcs_df.index.isin(route_set)]
        arc_mid_lons = []
        arc_mid_lats = []
        arc_ids_ordered = []
        for arc_id, arc_row in route_arcs.iterrows():
            lon_a, lat_a, lon_b, lat_b = _arc_endpoints(arc_row, nodes_df)
            arc_mid_lons.append((lon_a + lon_b) / 2)
            arc_mid_lats.append((lat_a + lat_b) / 2)
            arc_ids_ordered.append(arc_id)

        gt_per_obs = []
        for i in range(len(track)):
            best_arc = arc_ids_ordered[0]
            best_d = float("inf")
            for j, arc_id in enumerate(arc_ids_ordered):
                d = haversine(lats[i], lons[i], arc_mid_lats[j], arc_mid_lons[j])
                if d < best_d:
                    best_d = d
                    best_arc = arc_id
            gt_per_obs.append(best_arc)
    else:
        gt_per_obs = None

    # Main loop: generate candidates for each GPS observation
    for i, (_, gps_row) in enumerate(track.iterrows()):
        candidates_df = generate_candidates(
            gps_point=gps_row,
            nodes_df=nodes_df,
            arcs_df=arcs_df,
            k=k,
            gps_speed=float(speeds[i]),
            gps_heading=float(headings[i]),
            time_since_last=float(times_since[i]),
        )

        for _, cand_row in candidates_df.iterrows():
            feat_values = [cand_row[col] for col in FEATURE_COLUMNS]
            X_rows.append(feat_values)

            if has_labels and gt_per_obs is not None:
                label = int(int(cand_row["arc_id"]) == gt_per_obs[i])
                y_vals.append(label)

    X = np.array(X_rows, dtype=np.float64)
    y = np.array(y_vals, dtype=np.int32) if has_labels else None

    return X, y


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys, textwrap, io
    import pandas as pd

    # Minimal synthetic road network
    nodes_csv = textwrap.dedent("""\
        lon,lat
        -122.4194,37.7749
        -122.4185,37.7755
        -122.4175,37.7760
        -122.4165,37.7765
    """)
    arcs_csv = textwrap.dedent("""\
        from_node_id,to_node_id
        0,1
        1,2
        2,3
        0,2
    """)
    track_csv = textwrap.dedent("""\
        lon,lat,timestamp
        -122.4192,37.7750,1609459200
        -122.4188,37.7753,1609459201
        -122.4180,37.7757,1609459202
    """)
    route_txt = "0\n1\n2\n"

    nodes_df = pd.read_csv(io.StringIO(nodes_csv), index_col=None)
    nodes_df.index.name = "node_id"
    arcs_df = pd.read_csv(io.StringIO(arcs_csv), index_col=None)
    arcs_df.index.name = "arc_id"
    track_df = pd.read_csv(io.StringIO(track_csv))
    route = [int(x) for x in route_txt.strip().splitlines()]

    trip_dict = {
        "trip_id": "demo",
        "track": track_df,
        "nodes": nodes_df,
        "arcs": arcs_df,
        "route": route,
    }

    # Test generate_candidates
    gps_pt = track_df.iloc[0]
    cands = generate_candidates(gps_pt, nodes_df, arcs_df, k=3)
    print("=== generate_candidates (k=3) ===")
    print(cands[["arc_id", "perp_dist_m", "proj_ratio", "bearing_diff_deg"]].to_string(index=False))

    # Test engineer_features
    X, y = engineer_features(trip_dict, k=3)
    print(f"\n=== engineer_features ===")
    print(f"X shape : {X.shape}  (n_obs * k, {len(FEATURE_COLUMNS)} features)")
    print(f"y shape : {y.shape}")
    print(f"Positive labels: {y.sum()} / {len(y)}")
    print(f"\nFeature columns: {FEATURE_COLUMNS}")
    print("\nFirst row of X:")
    for col, val in zip(FEATURE_COLUMNS, X[0]):
        print(f"  {col:<22s}: {val:.4f}")

    # Test without route (evaluator trips)
    trip_no_route = {**trip_dict, "route": None}
    X2, y2 = engineer_features(trip_no_route, k=3)
    print(f"\nWithout route: X.shape={X2.shape}, y={y2}")

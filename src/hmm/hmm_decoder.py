"""
hmm_decoder.py
==============
HMM + Viterbi route reconstruction for GPS map-matching.

Theory
------
Given a sequence of GPS observations o_1, o_2, ..., o_T, and a road network
G = (V, E) where E are directed arcs, we model the hidden state at each step
as the arc the vehicle is travelling on.

The HMM has:
  - Hidden states  : arc IDs  (one per GPS step, chosen from top-k candidates)
  - Observations   : GPS points (lon, lat)
  - Emission prob  : P(o_t | arc_i)  ∝ classifier.predict_proba(features)
  - Transition prob: P(arc_j | arc_i) = exp(-beta * d(arc_i, arc_j))
      where d is the shortest path hop-count between the arcs in the road graph.
      d=0 (same arc)   → prob = 1.0
      d=1 (adjacent)   → prob = exp(-beta)
      disconnected     → prob = epsilon (very small)

Viterbi recurrence (log-domain):
  delta[t][j] = max_i { delta[t-1][i] + log P(arc_j | arc_i) } + log P(o_t | arc_j)
  psi[t][j]   = argmax_i { delta[t-1][i] + log P(arc_j | arc_i) }

Back-track from argmax_j delta[T][j] to recover the optimal state sequence.
"""

from __future__ import annotations

import json
import math
import pickle
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_LOG_ZERO = -1e18          # stand-in for log(0) — negative infinity
_EPSILON_PROB = 1e-10      # floor probability for disconnected arc pairs
_MAX_HOP_SEARCH = 5        # BFS depth cap for transition look-ups
_GPS_GAP_THRESHOLD = 60.0  # seconds — reset HMM across large GPS gaps


# ---------------------------------------------------------------------------
# Helper: arc-level graph
# ---------------------------------------------------------------------------

def _build_arc_graph(arcs_df: pd.DataFrame) -> nx.DiGraph:
    """
    Build a *node-level* directed graph from the arcs DataFrame.

    Nodes  → intersection IDs (integers from nodes file, NOT arc IDs)
    Edges  → road arcs, with arc_id stored as edge attribute

    This graph is used for shortest-path queries between arcs.
    """
    G = nx.DiGraph()
    for arc_id, row in arcs_df.iterrows():
        fn = int(row["from_node_id"])
        tn = int(row["to_node_id"])
        G.add_edge(fn, tn, arc_id=arc_id)
    return G


def _arc_endpoints(arc_id: int, arcs_df: pd.DataFrame) -> Tuple[int, int]:
    """Return (from_node_id, to_node_id) for a given arc_id."""
    row = arcs_df.loc[arc_id]
    return int(row["from_node_id"]), int(row["to_node_id"])


def _hop_distance(
    arc_i: int,
    arc_j: int,
    arcs_df: pd.DataFrame,
    G: nx.DiGraph,
    max_hops: int = _MAX_HOP_SEARCH,
) -> Optional[int]:
    """
    Approximate hop-distance between two arcs in the road graph.

    Strategy
    --------
    - If arc_i == arc_j            → 0
    - If arc_i.to_node == arc_j.from_node → 1  (direct connection)
    - Otherwise run BFS from arc_i.to_node toward arc_j.from_node
      (or arc_j.to_node for undirected fallback) up to max_hops.
    - Returns None if unreachable within max_hops.
    """
    if arc_i == arc_j:
        return 0

    fn_i, tn_i = _arc_endpoints(arc_i, arcs_df)
    fn_j, tn_j = _arc_endpoints(arc_j, arcs_df)

    # Direct connection
    if tn_i == fn_j:
        return 1

    # Shared node (undirected adjacency)
    nodes_i = {fn_i, tn_i}
    nodes_j = {fn_j, tn_j}
    if nodes_i & nodes_j:
        return 1

    # BFS in directed graph from tn_i to fn_j
    if G.has_node(tn_i) and G.has_node(fn_j):
        try:
            path = nx.shortest_path(G, source=tn_i, target=fn_j)
            hops = len(path) - 1  # number of edges traversed
            if hops <= max_hops:
                return hops
        except nx.NetworkXNoPath:
            pass

    # Undirected fallback: try reaching any node of arc_j from any node of arc_i
    if not G.is_directed():
        return None
    UG = G.to_undirected()
    for s in nodes_i:
        for t in nodes_j:
            if UG.has_node(s) and UG.has_node(t):
                try:
                    path = nx.shortest_path(UG, source=s, target=t)
                    hops = len(path) - 1
                    if hops <= max_hops:
                        return hops
                except nx.NetworkXNoPath:
                    continue

    return None  # disconnected within search radius


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class HMMDecoder:
    """
    HMM + Viterbi decoder for GPS map-matching.

    Parameters
    ----------
    nodes_df : pd.DataFrame
        Road-network nodes. Columns: lon, lat.  Index: node_id.
    arcs_df : pd.DataFrame
        Road-network arcs. Columns: from_node_id, to_node_id.  Index: arc_id.
    transition_beta : float, optional
        Decay rate for the transition probability:
            P(arc_j | arc_i) = exp(-beta * hop_distance(i, j))
        Larger beta → sharper preference for locally adjacent arcs.
        Default 0.1.
    gps_gap_threshold : float, optional
        Seconds.  GPS gaps larger than this trigger an HMM state reset
        (the previous path state is discarded).  Default 60.
    """

    def __init__(
        self,
        nodes_df: pd.DataFrame,
        arcs_df: pd.DataFrame,
        transition_beta: float = 0.1,
        gps_gap_threshold: float = _GPS_GAP_THRESHOLD,
    ):
        self.nodes_df = nodes_df.copy()
        self.arcs_df = arcs_df.copy()
        self.transition_beta = transition_beta
        self.gps_gap_threshold = gps_gap_threshold

        # Build node-level directed road graph
        self.G: nx.DiGraph = _build_arc_graph(arcs_df)

        # Cache for transition log-probs (arc_i, arc_j) → log_prob
        self._trans_cache: Dict[Tuple[int, int], float] = {}

    # ------------------------------------------------------------------
    # Transition probability helpers
    # ------------------------------------------------------------------

    def _transition_log_prob(self, arc_i: int, arc_j: int) -> float:
        """
        Compute log P(arc_j | arc_i) using hop-count exponential decay.

        Returns log(1.0) = 0.0  for directly connected arcs.
        Returns log(eps)        for topologically disconnected arcs.
        Uses a cache to avoid repeated BFS calls.
        """
        key = (arc_i, arc_j)
        if key in self._trans_cache:
            return self._trans_cache[key]

        hops = _hop_distance(arc_i, arc_j, self.arcs_df, self.G)
        if hops is None:
            # Disconnected within search radius
            lp = math.log(_EPSILON_PROB)
        else:
            # exp(-beta * hops); hops=0 → log_prob=0, hops=1 → -beta, ...
            lp = -self.transition_beta * hops

        self._trans_cache[key] = lp
        return lp

    # ------------------------------------------------------------------
    # Build transition matrix for a window of steps
    # ------------------------------------------------------------------

    def build_transition_matrix(
        self, candidate_arcs_per_step: List[List[int]]
    ) -> List[Dict[Tuple[int, int], float]]:
        """
        Pre-compute all pairwise transition log-probs for consecutive steps.

        Parameters
        ----------
        candidate_arcs_per_step : list of lists
            candidate_arcs_per_step[t] = list of arc_ids for GPS step t.

        Returns
        -------
        list of dicts, length = T-1.
            trans_matrices[t][(arc_i, arc_j)] = log P(arc_j | arc_i)
            for arc_i in candidates[t], arc_j in candidates[t+1].
        """
        T = len(candidate_arcs_per_step)
        matrices = []
        for t in range(T - 1):
            mat: Dict[Tuple[int, int], float] = {}
            for ai in candidate_arcs_per_step[t]:
                for aj in candidate_arcs_per_step[t + 1]:
                    mat[(ai, aj)] = self._transition_log_prob(ai, aj)
            matrices.append(mat)
        return matrices

    # ------------------------------------------------------------------
    # Fallback: nearest arc by node proximity
    # ------------------------------------------------------------------

    def _nearest_arc(self, lon: float, lat: float) -> int:
        """
        Find the arc whose midpoint is nearest to the given GPS coordinate.
        Used when no candidates are available for a step.
        """
        # Compute midpoint lon/lat for every arc
        from_lons = self.nodes_df.loc[self.arcs_df["from_node_id"], "lon"].values
        from_lats = self.nodes_df.loc[self.arcs_df["from_node_id"], "lat"].values
        to_lons   = self.nodes_df.loc[self.arcs_df["to_node_id"],   "lon"].values
        to_lats   = self.nodes_df.loc[self.arcs_df["to_node_id"],   "lat"].values
        mid_lons = (from_lons + to_lons) / 2
        mid_lats = (from_lats + to_lats) / 2

        # Euclidean on degree coords is a rough proxy — good enough for fallback
        dists = (mid_lons - lon) ** 2 + (mid_lats - lat) ** 2
        return int(self.arcs_df.index[np.argmin(dists)])

    # ------------------------------------------------------------------
    # Viterbi decoding
    # ------------------------------------------------------------------

    def viterbi_decode(
        self,
        emission_log_probs: List[Dict[int, float]],
        candidate_arcs_per_step: List[List[int]],
        timestamps: Optional[List[float]] = None,
    ) -> List[int]:
        """
        Run Viterbi decoding in log-space.

        Parameters
        ----------
        emission_log_probs : list of dicts
            emission_log_probs[t] = {arc_id: log P(o_t | arc_id)}
            for each candidate arc at GPS step t.
        candidate_arcs_per_step : list of lists
            candidate_arcs_per_step[t] = [arc_id, ...]  — ordered list.
        timestamps : list of float or None
            GPS timestamps (seconds).  Used to detect large gaps and reset
            the HMM state.  Pass None to skip gap detection.

        Returns
        -------
        list of int
            One arc_id per GPS step — the Viterbi-optimal route.

        Algorithm Notes
        ---------------
        Viterbi recurrence (all quantities in log-space):
          Initialisation  (t=0):
            delta[0][j] = log P(o_0 | arc_j)   # uniform prior over candidates
          Recursion       (t=1..T-1):
            delta[t][j] = max_i { delta[t-1][i] + log P(arc_j | arc_i) }
                          + log P(o_t | arc_j)
            psi[t][j]   = argmax_i { delta[t-1][i] + log P(arc_j | arc_i) }
          Termination:
            best_last   = argmax_j delta[T-1][j]
          Back-tracking:
            path[t]     = psi[t+1][path[t+1]]   for t = T-2 .. 0
        """
        T = len(emission_log_probs)
        if T == 0:
            return []

        predicted: List[int] = []
        # Segment the sequence at large GPS gaps
        segments = self._split_segments(T, timestamps)

        for seg_start, seg_end in segments:
            seg_pred = self._viterbi_segment(
                emission_log_probs[seg_start:seg_end],
                candidate_arcs_per_step[seg_start:seg_end],
            )
            predicted.extend(seg_pred)

        return predicted

    def _split_segments(
        self, T: int, timestamps: Optional[List[float]]
    ) -> List[Tuple[int, int]]:
        """
        Split a GPS track into segments at large time gaps.

        Returns list of (start, end) index pairs (end exclusive).
        """
        if timestamps is None or len(timestamps) < 2:
            return [(0, T)]

        segments = []
        seg_start = 0
        for t in range(1, T):
            gap = abs(timestamps[t] - timestamps[t - 1])
            if gap > self.gps_gap_threshold:
                segments.append((seg_start, t))
                seg_start = t
        segments.append((seg_start, T))
        return segments

    def _viterbi_segment(
        self,
        emission_log_probs: List[Dict[int, float]],
        candidate_arcs_per_step: List[List[int]],
    ) -> List[int]:
        """
        Run Viterbi on a single continuous segment (no GPS gaps).

        Returns the best arc_id list for the segment.
        """
        T = len(emission_log_probs)
        if T == 0:
            return []

        # ----------------------------------------------------------------
        # Initialisation — t = 0
        # ----------------------------------------------------------------
        candidates_0 = candidate_arcs_per_step[0]
        if not candidates_0:
            # No candidates: return most common arc overall as a fallback
            warnings.warn("No candidates at step 0; using fallback arc.")
            all_arcs = [a for cands in candidate_arcs_per_step for a in cands]
            fallback = int(max(set(all_arcs), key=all_arcs.count)) if all_arcs else 0
            return [fallback] * T

        # delta[j] = log P(path ending at arc j up to current step)
        # psi[t][j] = backpointer: which arc at t-1 led to arc j at t
        delta: Dict[int, float] = {}
        for arc in candidates_0:
            delta[arc] = emission_log_probs[0].get(arc, _LOG_ZERO)

        # psi[t] is a dict: arc_j → best predecessor arc_i at step t
        psi: List[Dict[int, int]] = [{}]  # psi[0] is unused (no predecessor)

        # ----------------------------------------------------------------
        # Recursion — t = 1 .. T-1
        # ----------------------------------------------------------------
        for t in range(1, T):
            new_delta: Dict[int, float] = {}
            new_psi: Dict[int, int] = {}

            candidates_t = candidate_arcs_per_step[t]
            if not candidates_t:
                # Fallback: extend the current best arc
                best_prev = max(delta, key=delta.get)
                new_delta = {best_prev: delta[best_prev]}
                new_psi   = {best_prev: best_prev}
            else:
                prev_arcs = list(delta.keys())
                for arc_j in candidates_t:
                    emit_lp = emission_log_probs[t].get(arc_j, _LOG_ZERO)
                    best_lp   = _LOG_ZERO
                    best_prev_arc = prev_arcs[0]

                    for arc_i in prev_arcs:
                        trans_lp = self._transition_log_prob(arc_i, arc_j)
                        score = delta[arc_i] + trans_lp
                        if score > best_lp:
                            best_lp       = score
                            best_prev_arc = arc_i

                    new_delta[arc_j] = best_lp + emit_lp
                    new_psi[arc_j]   = best_prev_arc

            delta = new_delta
            psi.append(new_psi)

        # ----------------------------------------------------------------
        # Termination — pick best arc at final step
        # ----------------------------------------------------------------
        if not delta:
            # Degenerate: return first candidate of each step
            return [
                (cands[0] if cands else 0)
                for cands in candidate_arcs_per_step
            ]

        best_last = max(delta, key=delta.get)

        # ----------------------------------------------------------------
        # Back-tracking
        # ----------------------------------------------------------------
        path = [best_last] * T
        for t in range(T - 2, -1, -1):
            path[t] = psi[t + 1].get(path[t + 1], path[t + 1])

        return path

    # ------------------------------------------------------------------
    # Full decode_trip pipeline
    # ------------------------------------------------------------------

    def decode_trip(
        self,
        trip_dict: Dict,
        classifier,
        feature_eng_fn,
        k: int = 5,
    ) -> List[int]:
        """
        End-to-end pipeline: GPS observations → predicted arc sequence.

        Parameters
        ----------
        trip_dict : dict
            Output of ``data_loader.load_trip()``.
            Must contain keys: 'track', 'nodes', 'arcs'.
        classifier : sklearn-compatible classifier
            Must have a ``predict_proba(X)`` method.
            Classes should correspond to arc_id binary label (0=wrong, 1=correct).
        feature_eng_fn : callable
            Signature: ``feature_eng_fn(gps_point, nodes_df, arcs_df, k)``
            Returns a DataFrame with one row per candidate arc and a column
            ``arc_id`` identifying which arc each row refers to.
            (Wraps ``generate_candidates`` from feature_engineering.)
        k : int
            Number of candidate arcs per GPS step.

        Returns
        -------
        list of int
            Predicted arc_id for each GPS observation.
        """
        track    = trip_dict["track"]
        nodes_df = trip_dict.get("nodes", self.nodes_df)
        arcs_df  = trip_dict.get("arcs",  self.arcs_df)

        emission_log_probs:       List[Dict[int, float]] = []
        candidate_arcs_per_step:  List[List[int]]        = []
        timestamps: List[float] = []

        for _, gps_row in track.iterrows():
            gps_point = gps_row  # pandas Series with lon, lat, timestamp

            # --- Generate candidate arcs ---
            try:
                candidates_df = feature_eng_fn(gps_point, nodes_df, arcs_df, k)
            except Exception:
                candidates_df = pd.DataFrame()

            if candidates_df is None or candidates_df.empty:
                # Fallback: use the nearest arc
                fallback_arc = self._nearest_arc(gps_point["lon"], gps_point["lat"])
                emission_log_probs.append({fallback_arc: 0.0})
                candidate_arcs_per_step.append([fallback_arc])
                timestamps.append(float(gps_row.get("timestamp", 0)))
                continue

            arc_ids = candidates_df["arc_id"].tolist() if "arc_id" in candidates_df.columns else list(candidates_df.index)

            # --- Get classifier probabilities ---
            feature_cols = [
                c for c in candidates_df.columns
                if c not in ("arc_id", "label", "trip_id")
            ]
            X_step = candidates_df[feature_cols].values

            try:
                proba = classifier.predict_proba(X_step)
                # proba shape: (n_candidates, n_classes)
                # We want P(correct arc) = proba[:, 1] if binary
                if proba.shape[1] == 2:
                    pos_proba = proba[:, 1]
                else:
                    pos_proba = proba[:, 0]
            except Exception:
                # Uniform if classifier fails
                pos_proba = np.ones(len(arc_ids)) / len(arc_ids)

            # Clip to avoid log(0)
            pos_proba = np.clip(pos_proba, _EPSILON_PROB, 1.0)
            log_probs_dict = {
                int(arc_id): float(np.log(p))
                for arc_id, p in zip(arc_ids, pos_proba)
            }

            emission_log_probs.append(log_probs_dict)
            candidate_arcs_per_step.append([int(a) for a in arc_ids])
            timestamps.append(float(gps_row.get("timestamp", 0)))

        return self.viterbi_decode(
            emission_log_probs,
            candidate_arcs_per_step,
            timestamps=timestamps,
        )

    # ------------------------------------------------------------------
    # Classifier-only baseline (for comparison)
    # ------------------------------------------------------------------

    def predict_classifier_only(
        self,
        trip_dict: Dict,
        classifier,
        feature_eng_fn,
        k: int = 5,
    ) -> List[int]:
        """
        Per-step argmax of classifier probability — no HMM smoothing.
        Used as a baseline to measure the benefit of Viterbi decoding.

        Returns
        -------
        list of int  — one arc_id per GPS step.
        """
        track    = trip_dict["track"]
        nodes_df = trip_dict.get("nodes", self.nodes_df)
        arcs_df  = trip_dict.get("arcs",  self.arcs_df)

        predicted: List[int] = []

        for _, gps_row in track.iterrows():
            try:
                candidates_df = feature_eng_fn(gps_row, nodes_df, arcs_df, k)
            except Exception:
                candidates_df = pd.DataFrame()

            if candidates_df is None or candidates_df.empty:
                predicted.append(self._nearest_arc(gps_row["lon"], gps_row["lat"]))
                continue

            arc_ids = candidates_df["arc_id"].tolist() if "arc_id" in candidates_df.columns else list(candidates_df.index)
            feature_cols = [
                c for c in candidates_df.columns
                if c not in ("arc_id", "label", "trip_id")
            ]
            X_step = candidates_df[feature_cols].values
            try:
                proba = classifier.predict_proba(X_step)
                pos_proba = proba[:, 1] if proba.shape[1] == 2 else proba[:, 0]
                best_idx = int(np.argmax(pos_proba))
            except Exception:
                best_idx = 0

            predicted.append(int(arc_ids[best_idx]))

        return predicted

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """
        Save the HMMDecoder (graph + beta + cache) to a pickle file.

        Parameters
        ----------
        path : str or Path
            Destination file path.  ``.pkl`` extension recommended.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(
                {
                    "nodes_df": self.nodes_df,
                    "arcs_df":  self.arcs_df,
                    "transition_beta": self.transition_beta,
                    "gps_gap_threshold": self.gps_gap_threshold,
                    "trans_cache": self._trans_cache,
                },
                fh,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

    @classmethod
    def load(cls, path: str | Path) -> "HMMDecoder":
        """
        Load a previously saved HMMDecoder from a pickle file.

        Parameters
        ----------
        path : str or Path
            Path to the ``.pkl`` file created by :meth:`save`.

        Returns
        -------
        HMMDecoder
        """
        path = Path(path)
        with open(path, "rb") as fh:
            data = pickle.load(fh)

        decoder = cls(
            nodes_df=data["nodes_df"],
            arcs_df=data["arcs_df"],
            transition_beta=data["transition_beta"],
            gps_gap_threshold=data.get("gps_gap_threshold", _GPS_GAP_THRESHOLD),
        )
        decoder._trans_cache = data.get("trans_cache", {})
        return decoder


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import textwrap, tempfile, os
    import sys

    print("Running HMMDecoder self-test with synthetic data …")

    # Minimal synthetic road network
    #
    #  node0 --arc0--> node1 --arc1--> node2
    #                   |
    #                  arc2
    #                   |
    #                  node3
    #
    nodes_df = pd.DataFrame(
        {"lon": [-122.419, -122.418, -122.417, -122.418],
         "lat": [ 37.774,   37.775,   37.776,   37.773]},
        index=pd.Index([0, 1, 2, 3], name="node_id"),
    )
    arcs_df = pd.DataFrame(
        {"from_node_id": [0, 1, 1],
         "to_node_id":   [1, 2, 3]},
        index=pd.Index([0, 1, 2], name="arc_id"),
    )

    decoder = HMMDecoder(nodes_df, arcs_df, transition_beta=0.5)

    # Three GPS steps, 2 candidates each
    candidates = [[0, 2], [0, 1], [1, 2]]
    emission_lp = [
        {0: -0.1, 2: -2.0},   # step 0: arc 0 most likely
        {0: -1.5, 1: -0.3},   # step 1: arc 1 most likely
        {1: -0.2, 2: -0.8},   # step 2: arc 1 or 2
    ]

    route = decoder.viterbi_decode(emission_lp, candidates)
    print(f"Viterbi route: {route}")
    assert len(route) == 3, "Route length mismatch"

    # Test save / load round-trip
    with tempfile.TemporaryDirectory() as tmp:
        pkl_path = os.path.join(tmp, "decoder.pkl")
        decoder.save(pkl_path)
        decoder2 = HMMDecoder.load(pkl_path)
        route2 = decoder2.viterbi_decode(emission_lp, candidates)
        assert route == route2, "Save/load round-trip mismatch"

    print("All self-tests passed.")

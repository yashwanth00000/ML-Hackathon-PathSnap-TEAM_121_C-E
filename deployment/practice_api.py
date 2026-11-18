from fastapi import FastAPI
from pathlib import Path
import numpy as np
import joblib

from src.preprocessing.data_loader import (
    load_track, load_nodes, load_arcs
)
from src.features.feature_engineering import (
    generate_candidates, haversine, compute_heading
)
from src.hmm.hmm_decoder import HMMDecoder

app = FastAPI(title="PathSnap Practice API")

model = joblib.load(
    "/tmp/pathsnap_practice_tuned.joblib"
)

folder = Path.home() / "Downloads/PathSnap_practice"

@app.get("/")
def home():
    return {
        "status": "running",
        "project": "PathSnap",
        "mode": "synthetic practice"
    }

@app.get("/trips")
def trips():
    return {"trips": ["demo"]}

@app.get("/predict/demo")
def predict():
    track = load_track(folder / "demo.track")
    nodes = load_nodes(folder / "demo.nodes")
    arcs = load_arcs(folder / "demo.arcs")

    decoder = HMMDecoder(
        nodes, arcs, transition_beta=0.5
    )

    candidates = []
    emissions = []
    baseline = []

    columns = [
        "perp_dist_m",
        "bearing_diff_deg",
        "gps_speed_mps",
        "arc_length_m"
    ]

    for i, row in track.iterrows():
        speed = 0.0
        heading = float("nan")
        dt = 0.0

        if i > 0:
            prev = track.iloc[i - 1]
            dt = float(row.timestamp - prev.timestamp)

            if dt > 0:
                speed = haversine(
                    prev.lat, prev.lon,
                    row.lat, row.lon
                ) / dt

                heading = compute_heading(
                    prev.lat, prev.lon,
                    row.lat, row.lon
                )

        df = generate_candidates(
            row, nodes, arcs,
            k=5,
            gps_speed=speed,
            gps_heading=heading,
            time_since_last=dt
        )

        ids = df.arc_id.astype(int).tolist()
        X = df[columns].to_numpy()

        assert np.isfinite(X).all()

        prob = model.predict_proba(X)[:, 1]

        assert np.isfinite(prob).all()

        candidates.append(ids)

        emissions.append({
            arc: float(np.log(max(p, 1e-10)))
            for arc, p in zip(ids, prob)
        })

        baseline.append(ids[int(np.argmax(prob))])

    predicted = decoder.viterbi_decode(
        emissions,
        candidates,
        timestamps=track.timestamp.tolist()
    )

    def continuity(route):
        if len(route) < 2:
            return None

        valid = sum(
            a == b or
            arcs.loc[a, "to_node_id"] ==
            arcs.loc[b, "from_node_id"]
            for a, b in zip(route[:-1], route[1:])
        )

        return valid / (len(route) - 1)

    return {
        "trip_id": "demo",
        "mode": "practice",
        "gps_observations": len(track),
        "classifier_route": baseline,
        "predicted_route": predicted,
        "classifier_continuity": continuity(baseline),
        "hmm_continuity": continuity(predicted)
    }

from fastapi.responses import HTMLResponse
import folium

@app.get("/ui", response_class=HTMLResponse)
def dashboard():
    return Path(__file__).with_name(
        "practice_ui.html"
    ).read_text()

@app.get("/map/demo", response_class=HTMLResponse)
def route_map():
    result = predict()

    track = load_track(folder / "demo.track")
    nodes = load_nodes(folder / "demo.nodes")
    arcs = load_arcs(folder / "demo.arcs")

    m = folium.Map(
        location=[track.lat.mean(), track.lon.mean()],
        zoom_start=17
    )

    def draw(name, route, color, weight):
        layer = folium.FeatureGroup(name=name)

        for arc_id in dict.fromkeys(route):
            arc = arcs.loc[arc_id]
            a = nodes.loc[arc.from_node_id]
            b = nodes.loc[arc.to_node_id]

            folium.PolyLine(
                [[a.lat, a.lon], [b.lat, b.lon]],
                color=color,
                weight=weight
            ).add_to(layer)

        layer.add_to(m)

    draw("Road Network", arcs.index.tolist(), "gray", 3)
    draw("Classifier Route", result["classifier_route"], "red", 7)
    draw("HMM Route", result["predicted_route"], "green", 4)

    gps = folium.FeatureGroup(name="GPS Observations")

    for _, row in track.iterrows():
        folium.CircleMarker(
            [row.lat, row.lon],
            radius=5,
            color="blue",
            fill=True,
            fill_opacity=1
        ).add_to(gps)

    gps.add_to(m)

    folium.LayerControl().add_to(m)

    m.fit_bounds(track[["lat", "lon"]].values.tolist())

    return m.get_root().render()

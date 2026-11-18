from pathlib import Path
import numpy as np
import pandas as pd
import folium

root = Path.cwd()
out = root / "results/official_hmm"

pred = pd.read_csv(out / "student_predictions.csv")

trip = int(pred.trip_id.iloc[0])

folder = next(
    p for p in (root / "data/student_test").iterdir()
    if p.is_dir() and p.name.isdigit()
    and int(p.name) == trip
)

stem = folder.name

track = np.loadtxt(
    folder / f"{stem}.track", ndmin=2
)

nodes = np.loadtxt(
    folder / f"{stem}.nodes", ndmin=2
)

arcs = np.loadtxt(
    folder / f"{stem}.arcs",
    dtype=int, ndmin=2
)

reference = np.loadtxt(
    folder / f"{stem}.route",
    dtype=int, ndmin=1
)

track = track[
    np.isfinite(track).all(axis=1)
    & (np.abs(track[:, 0]) <= 180)
    & (np.abs(track[:, 1]) <= 90)
]

track = track[
    np.argsort(track[:, 2], kind="stable")
]

p = pred[pred.trip_id == trip].sort_values(
    "observation_index"
)

assert len(track) == len(p)

with np.load(root / "data/processed/student_test.npz") as cache:
    nearby = np.unique(
        cache["ids"][cache["group"] == trip]
    )

m = folium.Map(
    location=[
        track[:, 1].mean(),
        track[:, 0].mean()
    ],
    zoom_start=15
)

def draw(name, route, color, weight):
    layer = folium.FeatureGroup(name=name)

    for arc_id in dict.fromkeys(route):
        a, b = arcs[int(arc_id)]

        folium.PolyLine(
            [
                nodes[a][::-1].tolist(),
                nodes[b][::-1].tolist()
            ],
            color=color,
            weight=weight,
            opacity=0.85
        ).add_to(layer)

    layer.add_to(m)

draw("Nearby Road Network", nearby, "gray", 2)
draw("Reference Route", reference, "green", 10)
draw("Classifier Route", p.classifier_arc, "red", 6)
draw("HMM Viterbi Route", p.hmm_arc, "purple", 4)

gps = folium.FeatureGroup(name="Raw GPS")

for lon, lat, timestamp in track:
    folium.CircleMarker(
        [lat, lon],
        radius=2,
        color="blue",
        fill=True,
        fill_opacity=0.8
    ).add_to(gps)

gps.add_to(m)

folium.LayerControl().add_to(m)

m.fit_bounds(track[:, :2][:, ::-1].tolist())

m.save(str(out / "real_route_map.html"))

print("Trip:", trip)
print("GPS observations:", len(track))
print("REAL DATA MAP GENERATED")

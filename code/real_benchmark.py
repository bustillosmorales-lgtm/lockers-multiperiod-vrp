"""
Reproducible real-geography locker benchmark. Replaces the single proprietary
case study with public instances whose locker locations and inter-locker distances
are REAL:

  - locker coordinates: OpenStreetMap parcel lockers (amenity=parcel_locker), pulled
    live from Overpass for a named place;
  - distances: shortest-path DRIVING distance on the real OSM street network
    (OSMnx graph, edge length in metres), not Euclidean;
  - demand: seeded and documented (real origin-destination locker volumes are not
    public; Kandoo cannot supply publishable ones either).

The generated instance uses the exact schema of voi.make_instance, so the whole
existing pipeline (model_full, matheuristic, certify, VoI) runs on it unchanged.
Everything is reproducible from a place name and a seed; nothing is proprietary.

build_real_instance(place, n, tau, seed) -> instance dict (+ coords, meta)
"""
import math
import random

import osmnx as ox
import networkx as nx

ox.settings.log_console = False
ox.settings.use_cache = True

UNIT_M = 100.0   # distance unit = 100 m, so magnitudes match the synthetic [0,100] grid


def _haversine(a, b):
    R = 6371000.0
    lat1, lon1, lat2, lon2 = map(math.radians, [a[0], a[1], b[0], b[1]])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * R * math.asin(math.sqrt(h))


def _locker_points(place):
    gdf = ox.features_from_place(place, {"amenity": "parcel_locker"})
    pts = [(g.y, g.x) for g in gdf.geometry if g.geom_type == "Point"]
    return pts


def build_real_instance(place, n, tau, seed, C=2, H=40, dem_hi=3):
    """A compact real delivery zone: a seeded locker + its n-1 nearest real lockers,
    with real driving distances among them and a depot at the zone centroid."""
    rng = random.Random(seed)
    pts = _locker_points(place)
    if len(pts) < n + 5:
        raise ValueError(f"{place}: only {len(pts)} lockers, need >= {n+5}")

    # seeded compact cluster: random anchor, then its n-1 nearest lockers
    anchor = pts[rng.randrange(len(pts))]
    nearest = sorted(pts, key=lambda p: _haversine(anchor, p))[:n]
    lat_c = sum(p[0] for p in nearest) / n
    lon_c = sum(p[1] for p in nearest) / n
    radius = max(_haversine((lat_c, lon_c), p) for p in nearest) + 800  # margin, m

    # real drive network around the zone, largest strongly-connected component
    G = ox.graph_from_point((lat_c, lon_c), dist=radius, network_type="drive")
    G = ox.truncate.largest_component(G, strongly=True)

    # depot (node 0) at the centroid; lockers 1..n at their real coordinates
    coords = {0: (lat_c, lon_c)}
    for k, p in enumerate(nearest, start=1):
        coords[k] = p
    lats = [coords[i][0] for i in range(n + 1)]
    lons = [coords[i][1] for i in range(n + 1)]
    nodes = ox.distance.nearest_nodes(G, X=lons, Y=lats)   # snap all points to graph

    # pairwise shortest-path driving distance (metres) -> integer 100 m units
    d = {}
    for i in range(n + 1):
        lengths = nx.shortest_path_length(G, nodes[i], weight="length")
        for j in range(n + 1):
            if i == j:
                continue
            m = lengths.get(nodes[j])
            if m is None:                       # disconnected pair: fall back to haversine
                m = _haversine(coords[i], coords[j])
            d[(i, j)] = max(1, round(m / UNIT_M))

    # seeded demand, identical scheme to voi.make_instance (geography-independent)
    T = list(range(1, tau + 1))
    p = {t: {j: {k: (rng.randint(0, dem_hi) if j != k else 0)
                 for k in range(1, n + 1)} for j in range(1, n + 1)} for t in T}
    g = {t: {k: rng.randint(0, dem_hi) for k in range(1, n + 1)} for t in T}

    inst = {"n": n, "T": T, "d": d, "p": p, "g": g, "C": C, "Q": 999, "H": H}
    inst["coords"] = coords
    inst["meta"] = {"place": place, "n_lockers_available": len(pts),
                    "zone_radius_m": round(radius), "unit_m": UNIT_M,
                    "n_graph_nodes": G.number_of_nodes()}
    return inst


if __name__ == "__main__":
    inst = build_real_instance("Piaseczno, Poland", n=8, tau=2, seed=0)
    m = inst["meta"]
    print("place:", m["place"], "| lockers available:", m["n_lockers_available"],
          "| graph nodes:", m["n_graph_nodes"], "| zone radius m:", m["zone_radius_m"])
    dd = inst["d"]
    sample = sorted(dd.items())[:6]
    print("sample distances (100 m units):", sample)
    vals = list(dd.values())
    print(f"distance units: min {min(vals)} max {max(vals)} mean {sum(vals)/len(vals):.1f}")

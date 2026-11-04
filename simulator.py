"""Team simulator + baseline, from the 02_baseline_simulator notebook, as an importable module.

The logic is the notebook's (zone_active, km_to_zone, check_leg, leg_is_safe, simulate, nearest_neighbour).
The only change: loading the data is wrapped in load(), so Approach A / Approach B can do
    import simulator as sim
    sim.load("01_data_prep", "data/zones(augmented data)")
    sim.simulate(tour, start_min, waits, routes)
"""
import os, csv, math

START_TIME = "08:00"
SPEED_KMH = 60
SAMPLE_KM = 0.5
EARTH_RADIUS_KM = 6371.0
DETOUR_FACTOR = 1.3
CORRIDOR_TOTAL_KM = 126.0
R185_MAIN_KM = 217.0
CHANGKAT_JERING, GUA_MUSANG = (4.78, 100.78), (4.8833, 101.9667)

towns, dist, zones, idx, ROUTE4 = [], [], [], {}, []
n, start, GERIK, JELI = 0, 0, None, None


def to_min(hhmm):
    """'19:30' -> 1170 (minutes after midnight)"""
    h, m = hhmm.split(":")
    return int(h) * 60 + int(m)


def to_clock(minutes):
    day, m = divmod(int(round(minutes)), 1440)
    return f"{m // 60:02d}:{m % 60:02d}" + (f" (+{day}d)" if day else "")


def _ll(t):
    return (t["lat"], t["lon"])


def load(in_dir, zones_dir, start_town="Ipoh", speed_kmh=None, corridor_total_km=None, r185_main_km=None):
    """Load towns.csv + distances_km.csv from in_dir and zones.csv + zone_points.csv from zones_dir."""
    global towns, dist, zones, idx, ROUTE4, n, start, GERIK, JELI
    global SPEED_KMH, CORRIDOR_TOTAL_KM, R185_MAIN_KM
    if speed_kmh is not None: SPEED_KMH = speed_kmh
    if corridor_total_km is not None: CORRIDOR_TOTAL_KM = corridor_total_km
    if r185_main_km is not None: R185_MAIN_KM = r185_main_km

    towns = []
    with open(os.path.join(in_dir, "towns.csv"), encoding="utf-8") as f:
        for r in csv.DictReader(f):
            towns.append({"id": int(r["id"]), "name": r["name"], "state": r["state"],
                          "district": r["district"], "lat": float(r["latitude"]),
                          "lon": float(r["longitude"]), "population": int(r["population"])})
    n = len(towns)
    dist = []
    with open(os.path.join(in_dir, "distances_km.csv")) as f:
        rows = csv.reader(f)
        next(rows)                                  # header row
        for r in rows:
            dist.append([float(x) for x in r[1:]])  # first column is the id
    idx = {t["name"]: t["id"] for t in towns}
    start = idx[start_town]

    def read_csv(name):
        with open(os.path.join(zones_dir, name), encoding="utf-8") as f:
            return list(csv.DictReader(f))

    zones = []
    for r in read_csv("zones.csv"):
        zones.append({"id": r["zone_id"], "name": r["name"], "buffer_km": float(r["buffer_km"]),
                      "active": (r["active_start"], r["active_end"]),
                      "active_min": (to_min(r["active_start"]), to_min(r["active_end"])),
                      "points": []})
    by_id = {z["id"]: z for z in zones}
    pts_name = "zone_points.csv" if os.path.exists(os.path.join(zones_dir, "zone_points.csv")) else "zones_points.csv"
    for r in sorted(read_csv(pts_name), key=lambda r: (r["zone_id"], int(r["seq"]))):
        by_id[r["zone_id"]]["points"].append((float(r["lat"]), float(r["lon"]), float(r["road_km"])))

    GERIK, JELI = _ll(towns[idx["Gerik"]]), _ll(towns[idx["Kampung Jeli"]])
    ROUTE4 = [(GERIK[0], GERIK[1], 0.0)]
    ROUTE4 += sorted((p for z in zones for p in z["points"]), key=lambda p: p[2])
    ROUTE4 += [(JELI[0], JELI[1], CORRIDOR_TOTAL_KM)]
    return dict(towns=n, zones=len(zones), start=start)


# ------------------------------------------------------------------ roads
def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def road_km(a, b):
    """Straight line x 1.3, a and b are (lat, lon)"""
    return haversine_km(a[0], a[1], b[0], b[1]) * DETOUR_FACTOR


def road_path(i, j, route="R4"):
    """Path from town i to town j as [(lat, lon, km_from_start)]."""
    a, b = towns[i], towns[j]
    if a["state"] == b["state"]:
        return [(a["lat"], a["lon"], 0.0), (b["lat"], b["lon"], dist[i][j])]
    if a["state"] == "Kelantan":                       # build Perak -> Kelantan, then flip it
        p = road_path(j, i, route)
        total = p[-1][2]
        return [(lat, lon, total - km) for lat, lon, km in reversed(p)]
    if route == "R4":
        start_km = road_km(_ll(a), GERIK)
        path = [(a["lat"], a["lon"], 0.0)] + [(lat, lon, start_km + km) for lat, lon, km in ROUTE4]
    else:                                              # Route 185 detour
        k1 = road_km(_ll(a), CHANGKAT_JERING)
        path = [(a["lat"], a["lon"], 0.0), (*CHANGKAT_JERING, k1), (*GUA_MUSANG, k1 + R185_MAIN_KM)]
    last = path[-1]
    return path + [(b["lat"], b["lon"], last[2] + road_km(last[:2], _ll(b)))]


def position(path, km):
    """Where is the car after driving `km` along the path?"""
    for (la1, lo1, k1), (la2, lo2, k2) in zip(path, path[1:]):
        if km <= k2:
            f = 0 if k2 == k1 else (km - k1) / (k2 - k1)
            return la1 + f * (la2 - la1), lo1 + f * (lo2 - lo1)
    return path[-1][:2]


# ------------------------------------------------------------------ simulator
def zone_active(zone, t):
    """t = minutes since midnight of day 1 (can be > 1440)"""
    s, e = zone["active_min"]
    c = t % 1440
    return s <= c < e if s < e else (c >= s or c < e)   # second case wraps past midnight


def _xy(lat, lon, lat0):
    return (math.radians(lon) * EARTH_RADIUS_KM * math.cos(math.radians(lat0)),
            math.radians(lat) * EARTH_RADIUS_KM)


def km_to_zone(lat, lon, zone):
    px, py = _xy(lat, lon, lat)
    pts = [_xy(a, b, lat) for a, b, _ in zone["points"]]
    best = float("inf")
    for (ax, ay), (bx, by) in zip(pts, pts[1:]):
        dx, dy = bx - ax, by - ay
        L2 = dx * dx + dy * dy
        f = 0 if L2 == 0 else max(0, min(1, ((px - ax) * dx + (py - ay) * dy) / L2))
        best = min(best, math.hypot(px - (ax + f * dx), py - (ay + f * dy)))
    return best


def check_leg(i, j, depart, route="R4"):
    """Drive town i -> town j leaving at `depart`. Returns (arrive, km, list of zone ids hit while active)."""
    path = road_path(i, j, route)
    total_km = path[-1][2]
    steps = max(1, math.ceil(total_km / SAMPLE_KM))
    hits = []
    for k in range(steps + 1):
        km = total_km * k / steps
        lat, lon = position(path, km)
        t = depart + km / SPEED_KMH * 60                # clock time at this point
        for z in zones:
            if z["id"] not in hits and zone_active(z, t) and km_to_zone(lat, lon, z) <= z["buffer_km"]:
                hits.append(z["id"])
    return depart + total_km / SPEED_KMH * 60, total_km, hits


def leg_is_safe(i, j, depart, route="R4"):
    """True if leaving i for j at `depart` never enters an active zone."""
    return not check_leg(i, j, depart, route)[2]


def simulate(tour, start_min, waits=None, routes=None):
    """tour = [v0, ..., v0]. waits = {town_id: minutes}. routes = {leg_number: 'R185'}."""
    waits, routes = waits or {}, routes or {}
    t = start_min
    legs = []
    for k, (i, j) in enumerate(zip(tour, tour[1:])):
        w = waits.get(i, 0)
        depart = t + w
        route = routes.get(k, "R4") if towns[i]["state"] != towns[j]["state"] else "local"
        arrive, km, hits = check_leg(i, j, depart, route)
        legs.append({"from": i, "to": j, "route": route, "wait_min": w, "depart": depart,
                     "arrive": arrive, "km": km, "unsafe": bool(hits), "zones_hit": hits})
        t = arrive
    return {"legs": legs, "total_min": t - start_min, "total_km": sum(l["km"] for l in legs),
            "wait_min": sum(l["wait_min"] for l in legs),
            "unsafe_legs": sum(l["unsafe"] for l in legs)}


# ------------------------------------------------------------------ baseline
def nearest_neighbour(start_id=None):
    """Classic NN by distances_km.csv. Ignores the zones and the clock."""
    s = start if start_id is None else start_id
    route, current = [s], s
    unvisited = set(range(n)) - {s}
    while unvisited:
        nxt = min(sorted(unvisited), key=lambda c: dist[current][c])
        route.append(nxt)
        unvisited.remove(nxt)
        current = nxt
    route.append(s)
    return route

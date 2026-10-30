"""Approach A: time-aware Nearest Neighbour with a spatial grid index (CS201 G2T4).

Usage:  python approach_A_time_aware_nn.py [path/to/MY.txt] [folder with zones.csv and zone_points.csv]
Without MY.txt it falls back to a small approximate DEMO town set (not for reporting).
Tables are saved as CSV and plots as PNG in ./outputs
"""


# # Approach A: Time-Aware Nearest Neighbour with a Spatial Grid Index
# **CS201 G2T4: Time-aware touring around elephant crossing zones (Perak and Kelantan)**
# 
# What this notebook does
# 1. Loads towns (GeoNames `MY.txt`) and your four danger zones.
# 2. Builds a **grid index** over the zones.
# 3. Builds each leg as a path, samples points along it, and uses the grid to find which zones the leg touches. That result is cached as `(zone, entry_km, exit_km)`.
# 4. Runs **time-aware NN**: from the current town, for each unvisited town pick the cheapest of *direct*, *wait-then-direct* or *detour (Route 185)*, then go to the one with the **earliest arrival**.
# 5. Compares against a classic NN baseline, checks safety with an **independent brute-force verifier**, and runs the experiments.
# 
# Modelling assumptions (all editable in the config cell)
# - Same-region legs (Perak to Perak, Kelantan to Kelantan) are straight-line x 1.3 and never touch a zone.
# - Cross-region legs use **Route 4** (through Z1 to Z4) or the **Route 185 detour** via Gua Musang (no zones). Gateways: Gerik (Perak side), Jeli (Kelantan side).
# - One speed for everything, zone hours repeat daily, waiting only at towns.


import math, time, os, io, tracemalloc, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)
import matplotlib
if not os.environ.get("DISPLAY") and os.name != "nt": matplotlib.use("Agg")
def display(x):
    print(x.to_string() if hasattr(x, "to_string") else x)

CFG = dict(
    SPEED_KMH      = 60.0,   # default; run 50 as sensitivity
    DETOUR_FACTOR  = 1.3,    # haversine -> road km (off the corridor)
    SAMPLE_STEP_KM = 0.5,    # sampling spacing along a leg
    GRID_CELL_KM   = 5.0,
    N_TOWNS        = 60,     # total towns, split between Perak and Kelantan
    START_NAME     = "Ipoh",
    START_HOUR     = 8.0,
    # --- Route 4 corridor (Gerik -> Jeli), measured on Google Maps ---
    CORRIDOR_TOTAL_KM = 126.0,
    KM0_OFFSET_KM     = 0.0,   # distance from Gerik town to zone-table km 0. See the sanity-check cell.
    # --- Route 185 detour (no zones), measured: Perak end -> Gua Musang ---
    R185_MAIN_KM   = 217.0,
)

# Gateway points (APPROXIMATE: overridden from your dataset if found, please verify)
GATE = dict(
    GERIK=(5.4308, 101.1275),
    JELI=(5.7000, 101.8333),
    CJ=(4.7800, 100.7800),      # Changkat Jering (Perak end of Route 185)
    GM=(4.8833, 101.9667),      # Gua Musang
)

# Your zone table (times as H:MM)
ZONE_ROWS = [
 ("Z1","KM11 near Kuala Rui",11,9,13,5.46596,101.17158,1,"19:00","7:00","3 reported incidents (Jan 2021, 23 Oct 2024, 20 May 2025)"),
 ("Z2","KM31 near Tasik Banding",31,29,33,5.53333,101.26245,1,"19:00","7:00","Motorcyclist collided with elephant ~9pm (10 Dec 2025)"),
 ("Z3","KM50",50,48,52,5.56473,101.36287,1,"19:00","7:00","Calf killed by SUV (Apr 2024); no time in report"),
 ("Z4","KM80",80,78,82,5.59922,101.5784,1,"19:00","7:00","Calf killed by lorry ~2:50am (11 May 2025)"),
]
import sys
ZONES_DIR = sys.argv[2] if len(sys.argv) > 2 else "zones(augmented data)"
_zc, _zp = os.path.join(ZONES_DIR, "zones.csv"), os.path.join(ZONES_DIR, "zone_points.csv")
if os.path.exists(_zc):
    zones_df = pd.read_csv(_zc, sep=None, engine="python")
    print("Loaded", len(zones_df), "zones from", _zc)
else:
    print("zones.csv not found -> using the 4 built-in zones")
    zones_df = pd.DataFrame(ZONE_ROWS, columns=["zone_id","name","road_km","km_start","km_end",
                         "center_lat","center_lon","buffer_km","active_start","active_end","evidence"])
if os.path.exists(_zp):
    zone_pts = pd.read_csv(_zp, sep=None, engine="python")      # zone_id, seq, road_km, lat, lon
    print("Loaded", len(zone_pts), "road points from", _zp)
else:
    zone_pts = None
    print("zone_points.csv not found -> each zone is ONE disc at its centre (coarser geometry)")
def hhmm(s):
    h, m = s.strip().split(":"); return int(h)*60 + int(m)
zones_df["a_s"] = zones_df.active_start.map(hhmm)
zones_df["a_e"] = zones_df.active_end.map(hhmm)
# zone extent along the road (km) = half the stretch + buffer
zones_df["radius_road_km"] = zones_df.buffer_km + (zones_df.km_end - zones_df.km_start)/2
display(zones_df[["zone_id","road_km","center_lat","center_lon","radius_road_km","a_s","a_e"]])


# ## 1. Geometry helpers, time-window logic, and the grid index


LAT0, LON0 = 5.5, 101.5
KX = 111.32*math.cos(math.radians(LAT0)); KY = 110.57

def to_xy(lat, lon):
    return (np.asarray(lon)-LON0)*KX, (np.asarray(lat)-LAT0)*KY

def hav(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2-p1)/2)**2 + np.cos(p1)*np.cos(p2)*np.sin(np.radians(np.asarray(lon2)-lon1)/2)**2
    return 2*6371.0088*np.arcsin(np.sqrt(a))

def est(a, b):
    """Model road km between two (lat, lon): haversine x detour factor."""
    return float(hav(a[0], a[1], b[0], b[1]))*CFG["DETOUR_FACTOR"]

# ---------- active-window logic (handles wrap past midnight) ----------
def active_conflict(s, e, a_s, a_e):
    """If [s, e] (absolute minutes) overlaps a daily active window [a_s, a_e) return the END of that window, else None."""
    length = (a_e - a_s) % 1440 or 1440
    for k in range(math.floor((s - a_s)/1440) - 1, math.floor((e - a_s)/1440) + 2):
        ws = k*1440 + a_s; we = ws + length
        if s < we and e > ws:
            return we
    return None

# ---------- grid index over zones ----------
OFF, MUL = 20000, 100000
class ZoneIndex:
    """Zones are stored as DISCS. A zone with a road polyline becomes many discs (one per polyline point,
    radius = buffer); a zone without one is a single disc. The grid maps cell -> discs, hits are reported per ZONE."""
    def __init__(self, dx, dy, dr, dzone, a_s, a_e, cell_km):
        self.zx, self.zy, self.zr = (np.asarray(v, float) for v in (dx, dy, dr))
        self.dzone = np.asarray(dzone, int)
        self.a_s, self.a_e = np.asarray(a_s), np.asarray(a_e)                 # per zone
        self.da_s, self.da_e = self.a_s[self.dzone], self.a_e[self.dzone]     # per disc
        self.cell = cell_km
        self.grid = {}
        for d in range(len(self.zx)):
            cx0 = int(math.floor((self.zx[d]-self.zr[d])/cell_km)); cx1 = int(math.floor((self.zx[d]+self.zr[d])/cell_km))
            cy0 = int(math.floor((self.zy[d]-self.zr[d])/cell_km)); cy1 = int(math.floor((self.zy[d]+self.zr[d])/cell_km))
            for cx in range(cx0, cx1+1):
                for cy in range(cy0, cy1+1):
                    self.grid.setdefault((cx+OFF)*MUL + (cy+OFF), []).append(d)
        self.occ = np.array(sorted(self.grid), dtype=np.int64)

    def keys(self, x, y):
        cx = np.floor(x/self.cell).astype(np.int64); cy = np.floor(y/self.cell).astype(np.int64)
        return (cx+OFF)*MUL + (cy+OFF)

    def hits_grid(self, s, x, y, pad):
        """Only samples in an occupied grid cell are tested, and only against the discs registered in that cell."""
        keys = self.keys(x, y)
        cand = np.nonzero(np.isin(keys, self.occ))[0]
        found = {}
        for i in cand:
            for d in self.grid[int(keys[i])]:
                if (x[i]-self.zx[d])**2 + (y[i]-self.zy[d])**2 <= self.zr[d]**2:
                    z = int(self.dzone[d])
                    lo, hi = found.get(z, (s[i], s[i]))
                    found[z] = (min(lo, s[i]), max(hi, s[i]))
        return sorted((z, max(0.0, lo-pad), hi+pad) for z, (lo, hi) in found.items())

    def hits_brute(self, s, x, y, pad):
        """Reference: every sample vs every disc."""
        d2 = (x[:, None]-self.zx[None, :])**2 + (y[:, None]-self.zy[None, :])**2
        inside = d2 <= self.zr[None, :]**2
        out = []
        for z in np.unique(self.dzone):
            m = inside[:, self.dzone == z].any(axis=1)
            if m.any():
                out.append((int(z), max(0.0, s[m].min()-pad), s[m].max()+pad))
        return sorted(out)

def sample_path(path, step):
    lats, lons, cum = path
    s = np.append(np.arange(0, cum[-1], step), cum[-1])
    x, y = to_xy(np.interp(s, cum, lats), np.interp(s, cum, lons))
    return s, x, y


# ## 2. Load towns
# 
# Upload `MY.txt` when prompted in Colab (or put it next to the notebook). If it is not found, a **small approximate demo set** is used so the notebook still runs. Do not report results from the demo set.
# 
# The Perak and Kelantan `admin1` codes are **auto-detected** from anchor towns (Ipoh, Kota Bharu) instead of being hard-coded, because the code in the brief should be double-checked. Verify the printout.


GEO_COLS = ["geonameid","name","asciiname","alternatenames","lat","lon","fclass","fcode","cc","cc2",
            "admin1","admin2","admin3","admin4","population","elevation","dem","tz","moddate"]

def load_geonames(path):
    df = pd.read_csv(path, sep="\t", header=None, names=GEO_COLS, dtype={"admin1": str},
                     quoting=3, low_memory=False)
    return df[df.fclass == "P"].copy()

def code_of(df, nm):
    r = df[df.asciiname.str.lower() == nm.lower()]
    return r.admin1.mode().iat[0] if len(r) else None

DEMO = [  # APPROXIMATE coordinates, demo only
 ("Ipoh",4.5975,101.0901,"P",657892),("Taiping",4.85,100.74,"P",245000),("Kuala Kangsar",4.7667,100.9333,"P",40000),
 ("Gerik",5.4308,101.1275,"P",38000),("Teluk Intan",4.0259,101.0213,"P",100000),("Batu Gajah",4.4719,101.0419,"P",40000),
 ("Kota Bharu",6.1333,102.25,"K",491000),("Bachok",6.0667,102.4,"K",30000),("Machang",5.7667,102.2167,"K",60000),
 ("Tanah Merah",5.8,102.15,"K",40000),("Pasir Mas",6.05,102.14,"K",50000),("Kuala Krai",5.5333,102.2,"K",30000),
 ("Gua Musang",4.8833,101.9667,"K",40000),("Kampung Jeli",5.70,101.8333,"K",5000)]

import sys
MY_PATH = sys.argv[1] if len(sys.argv) > 1 else "MY.txt"
if not os.path.exists(MY_PATH):
    print("MY.txt not found at", MY_PATH, "-> using DEMO towns (approximate, not for reporting).")
    MY_PATH = None

if MY_PATH:
    raw = load_geonames(MY_PATH)
    pc, kc = code_of(raw, "Ipoh"), code_of(raw, "Kota Bharu")
    print("Detected admin1 codes -> Perak:", pc, " Kelantan:", kc, "(verify against admin1CodesASCII.txt)")
    raw = raw[raw.admin1.isin([pc, kc])].copy()
    raw["region"] = np.where(raw.admin1 == pc, "P", "K")
    # override gateway coordinates from dataset where possible
    for key, nm, reg in [("GERIK","Gerik","P"), ("JELI","Kampung Jeli","K"), ("GM","Gua Musang","K")]:
        r = raw[(raw.asciiname == nm) & (raw.region == reg)]
        if len(r): GATE[key] = (float(r.lat.iloc[0]), float(r.lon.iloc[0])); print("gateway", key, "from dataset:", GATE[key])
    half = CFG["N_TOWNS"]//2
    towns = pd.concat([g.sort_values("population", ascending=False).head(half) for _, g in raw.groupby("region")])
    towns = towns.rename(columns={"asciiname": "town"})[["town","lat","lon","region","population"]]
else:
    towns = pd.DataFrame(DEMO, columns=["town","lat","lon","region","population"])
    GATE["GERIK"] = (5.4308, 101.1275); GATE["JELI"] = (5.70, 101.8333)

if CFG["START_NAME"] not in set(towns.town):
    s0 = raw[raw.asciiname == CFG["START_NAME"]] if MY_PATH else None
    if s0 is not None and len(s0):
        r = s0.iloc[0]
        towns = pd.concat([towns, pd.DataFrame([dict(town=CFG["START_NAME"], lat=r.lat, lon=r.lon, region=r.region, population=r.population)])])
towns = towns.reset_index(drop=True)
print(towns.region.value_counts().to_dict(), "towns;", len(towns), "total")
display(towns.head())


# ## 3. Corridor sanity check (km 0)
# 
# The zone table gives `road_km`, but not where km 0 is. This check puts the zones on a polyline from Gerik to Jeli and prints the implied **road km / straight-line km** for each stretch. A winding hill road should give roughly 1.2 to 1.7. A ratio far outside that means the km 0 assumption is wrong. Try `KM0_OFFSET_KM` = 0 and 12.5 and compare.


Z = zones_df.sort_values("road_km").reset_index(drop=True)

def corridor_knots(km0):
    """[(lat, lon, road_km_from_Gerik)] for Gerik, Z1..Z4, Jeli."""
    k = [(GATE["GERIK"][0], GATE["GERIK"][1], 0.0)]
    if zone_pts is not None:
        k += [(r.lat, r.lon, km0 + r.road_km) for r in zone_pts.sort_values("road_km").itertuples()]
    else:
        k += [(r.center_lat, r.center_lon, km0 + r.road_km) for r in Z.itertuples()]
    k += [(GATE["JELI"][0], GATE["JELI"][1], CFG["CORRIDOR_TOTAL_KM"])]
    return k

for km0 in (0.0, 12.5):
    ks = corridor_knots(km0); rows = []
    for a, b in zip(ks[:-1], ks[1:]):
        r = b[2]-a[2]
        if r < 2.0: continue                      # skip points inside a zone; show only the gaps between zones
        g = float(hav(a[0], a[1], b[0], b[1]))
        rows.append(dict(km0=km0, road_km=round(r,1), straight_km=round(g,1), ratio=round(r/g,2)))
    print(pd.DataFrame(rows).to_string(index=False)); print()


# ## 4. The world: distance matrix, leg paths, cached zone hits, and options


def is_safe(hits, D, zidx, speed):
    return all(active_conflict(D + a/speed*60, D + b/speed*60, zidx.a_s[z], zidx.a_e[z]) is None for z, a, b in hits)

def earliest_safe(hits, D, zidx, speed, limit=20):
    """Earliest departure >= D at which every zone hit falls outside its active window (waiting only at the town)."""
    for _ in range(limit):
        nxt = None
        for z, a, b in hits:
            we = active_conflict(D + a/speed*60, D + b/speed*60, zidx.a_s[z], zidx.a_e[z])
            if we is not None:
                cand = we - a/speed*60          # enter the zone exactly when its window closes
                nxt = cand if nxt is None else max(nxt, cand)
        if nxt is None: return D
        D = nxt
    return None

def r4_path_east(p, k):
    """Perak town p -> Gerik -> Z1..Z4 -> Jeli -> Kelantan town k.  p, k are (lat, lon)."""
    ks = corridor_knots(CFG["KM0_OFFSET_KM"]); base = est(p, GATE["GERIK"])
    lats = [p[0]] + [q[0] for q in ks] + [k[0]]
    lons = [p[1]] + [q[1] for q in ks] + [k[1]]
    cum = [0.0] + [base + q[2] for q in ks]; cum.append(cum[-1] + est(GATE["JELI"], k))
    return _fix(lats, lons, cum)

def r185_path_east(p, k):
    lats = [p[0], GATE["CJ"][0], GATE["GM"][0], k[0]]; lons = [p[1], GATE["CJ"][1], GATE["GM"][1], k[1]]
    c1 = est(p, GATE["CJ"]); c2 = c1 + CFG["R185_MAIN_KM"]; c3 = c2 + est(GATE["GM"], k)
    return _fix(lats, lons, [0.0, c1, c2, c3])

def _fix(lats, lons, cum):
    cum = np.array(cum, float) + np.arange(len(cum))*1e-9     # strictly increasing for np.interp
    return np.array(lats), np.array(lons), cum

def reverse(path):
    la, lo, c = path; return la[::-1], lo[::-1], c[-1] - c[::-1]

class World:
    def __init__(self, towns, zones, cell_km=None):
        self.towns = towns; self.n = len(towns)
        self.name = towns.town.tolist(); self.region = towns.region.values
        self.ll = list(zip(towns.lat, towns.lon))
        la, lo = towns.lat.values, towns.lon.values
        self.D = hav(la[:, None], lo[:, None], la[None, :], lo[None, :]) * CFG["DETOUR_FACTOR"]
        self.speed = CFG["SPEED_KMH"]; self.step = CFG["SAMPLE_STEP_KM"]
        # convert zone radius (road km) to map km using the corridor winding ratio
        ks = corridor_knots(CFG["KM0_OFFSET_KM"])
        geo = sum(float(hav(a[0], a[1], b[0], b[1])) for a, b in zip(ks[:-1], ks[1:]))
        self.winding = CFG["CORRIDOR_TOTAL_KM"] / geo
        self.zones = zones.reset_index(drop=True)
        dx, dy, dr, dz = [], [], [], []
        for zi, r in enumerate(self.zones.itertuples()):
            pts = zone_pts[zone_pts.zone_id == r.zone_id] if zone_pts is not None else None
            if pts is not None and len(pts):          # real road geometry: a disc of radius=buffer at every road point
                x, y = to_xy(pts.lat.values, pts.lon.values); rad = np.full(len(pts), float(r.buffer_km))
            else:                                      # fallback: one disc at the zone centre
                x, y = to_xy([r.center_lat], [r.center_lon]); rad = np.array([r.radius_road_km/self.winding])
            dx += list(x); dy += list(y); dr += list(rad); dz += [zi]*len(rad)
        self.zidx = ZoneIndex(dx, dy, dr, dz, self.zones.a_s.values, self.zones.a_e.values, cell_km or CFG["GRID_CELL_KM"])
        self._path, self._hits = {}, {}

    def path(self, i, j, route):
        key = (i, j, route)
        if key not in self._path:
            f = r4_path_east if route == "R4" else r185_path_east
            self._path[key] = f(self.ll[i], self.ll[j]) if self.region[i] == "P" else reverse(f(self.ll[j], self.ll[i]))
        return self._path[key]

    def km(self, i, j, route): return float(self.path(i, j, route)[2][-1])

    def hits(self, i, j, route):
        key = (i, j, route)
        if key not in self._hits:
            s, x, y = sample_path(self.path(i, j, route), self.step)
            self._hits[key] = self.zidx.hits_grid(s, x, y, pad=self.step)
        return self._hits[key]

    def best_option(self, i, j, t, mode="full"):
        """mode: 'full' (wait or detour), 'wait_only', 'detour_only'."""
        sp = self.speed
        if self.region[i] == self.region[j]:
            km = self.D[i, j]
            return dict(route="local", type="direct", depart=t, wait=0.0, arrive=t + km/sp*60, km=km, km_r4=km, r4_arrive=None, r185_arrive=None)
        km4, km5 = self.km(i, j, "R4"), self.km(i, j, "R185")
        D4 = earliest_safe(self.hits(i, j, "R4"), t, self.zidx, sp)
        r4_arr = (D4 + km4/sp*60) if D4 is not None else float("inf")
        r185_arr = t + km5/sp*60
        cands = []
        if D4 == t: cands.append(dict(route="R4", type="direct", depart=t, arrive=r4_arr, km=km4))
        elif mode in ("full", "wait_only") and D4 is not None: cands.append(dict(route="R4", type="wait-then-direct", depart=D4, arrive=r4_arr, km=km4))
        if mode in ("full", "detour_only"): cands.append(dict(route="R185", type="detour", depart=t, arrive=r185_arr, km=km5))
        b = min(cands, key=lambda c: c["arrive"])      # ties keep the earlier (direct first)
        b.update(wait=b["depart"] - t, km_r4=km4, r4_arrive=r4_arr, r185_arrive=r185_arr)
        return b


# ## 5. Baseline NN, Approach A, and the independent safety verifier


def run_baseline_nn(w, start, t0):
    """Classic NN: picks the nearest town by distance, ignores zones, never waits.
    Cross-region legs are driven on Route 4 (the only road it knows)."""
    unv = set(range(w.n)) - {start}; cur, t, legs = start, t0, []
    def go(a, b):
        nonlocal t
        if w.region[a] == w.region[b]: km, route = w.D[a, b], "local"
        else: km, route = w.km(a, b, "R4"), "R4"
        arr = t + km/w.speed*60
        legs.append(dict(frm=a, to=b, depart=t, wait=0.0, arrive=arr, type="direct", route=route, km=km, km_r4=km))
        t = arr
    while unv:
        nxt = min(sorted(unv), key=lambda u: w.D[cur, u]); go(cur, nxt); unv.remove(nxt); cur = nxt
    go(cur, start)
    return legs

def run_time_aware_nn(w, start, t0, mode="full"):
    unv = set(range(w.n)) - {start}; cur, t, legs = start, t0, []
    while unv:
        best, bu = None, None
        for u in sorted(unv):
            o = w.best_option(cur, u, t, mode)
            if best is None or o["arrive"] < best["arrive"] - 1e-9: best, bu = o, u
        best.update(frm=cur, to=bu); legs.append(best)
        t = best["arrive"]; unv.remove(bu); cur = bu
    o = w.best_option(cur, start, t, mode); o.update(frm=cur, to=start); legs.append(o)
    return legs

def verify(w, legs, step=0.1):
    """Independent check: re-sample every leg at a finer step, test EVERY sample against EVERY zone
    (no grid, no cache), using the sample's own clock time. Returns (unsafe legs, unsafe samples)."""
    zx, zy, zr, a_s, a_e = w.zidx.zx, w.zidx.zy, w.zidx.zr, w.zidx.da_s, w.zidx.da_e
    bad_legs = bad_samples = 0
    for L in legs:
        if L["route"] == "local": continue
        s, x, y = sample_path(w.path(L["frm"], L["to"], L["route"]), step)
        tm = (L["depart"] + s/w.speed*60) % 1440
        inside = ((x[:, None]-zx)**2 + (y[:, None]-zy)**2) <= zr**2
        m = tm[:, None]          # (each unsafe SAMPLE is counted once below, even if it lies in several discs)
        active = np.where(a_s < a_e, (m >= a_s) & (m < a_e), (m >= a_s) | (m < a_e))
        n = int((inside & active).any(axis=1).sum()); bad_samples += n; bad_legs += n > 0
    return bad_legs, bad_samples

def summarize(label, legs, t0, w):
    ul, us = verify(w, legs)
    return dict(method=label, total_h=round((legs[-1]["arrive"]-t0)/60, 2),
                wait_h=round(sum(l["wait"] for l in legs)/60, 2),
                n_waits=sum(l["wait"] > 1e-9 for l in legs),
                n_detours=sum(l["type"] == "detour" for l in legs),
                detour_extra_km=round(sum(l["km"]-l["km_r4"] for l in legs if l["type"] == "detour"), 1),
                total_km=round(sum(l["km"] for l in legs), 1),
                unsafe_legs=ul, unsafe_samples=us)

W = World(towns, zones_df)
START = W.name.index(CFG["START_NAME"]) if CFG["START_NAME"] in W.name else 0
print("start:", W.name[START], "| towns:", W.n, "| corridor winding ratio:", round(W.winding, 2),
      "| zone discs in grid:", len(W.zidx.zr))


# ## Experiment 1: Baseline vs Approach A, at three start times
# `unsafe_legs` and `unsafe_samples` come from the independent brute-force verifier. Approach A must show **0**.


def compare(w, start_hour):
    t0 = start_hour*60
    rows = [summarize("Baseline NN (ignores zones)", run_baseline_nn(w, START, t0), t0, w)]
    for mode, lab in [("wait_only", "A: wait only"), ("detour_only", "A: detour only"), ("full", "A: wait + detour")]:
        rows.append(summarize(lab, run_time_aware_nn(w, START, t0, mode), t0, w))
    df = pd.DataFrame(rows); df.insert(0, "start", f"{int(start_hour):02d}:00"); return df

exp1 = pd.concat([compare(W, h) for h in (6, 8, 18)], ignore_index=True)
display(exp1)
exp1.to_csv(os.path.join(OUT_DIR, "exp1_baseline_vs_A.csv"), index=False)
assert (exp1[exp1.method.str.startswith("A:")].unsafe_samples == 0).all(), "Approach A produced an unsafe crossing!"


# ## Experiment 2: Sensitivity to start time and speed


hours = list(range(24)); res = {}
for sp in (60.0, 50.0):
    W.speed = sp
    for mode in ("wait_only", "detour_only", "full"):
        res[(sp, mode)] = [(run_time_aware_nn(W, START, h*60, mode)[-1]["arrive"] - h*60)/60 for h in hours]
W.speed = CFG["SPEED_KMH"]

fig, ax = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
for a, sp in zip(ax, (60.0, 50.0)):
    for mode in ("wait_only", "detour_only", "full"): a.plot(hours, res[(sp, mode)], marker="o", ms=3, label=mode)
    a.set_title(f"Total trip time vs start hour ({int(sp)} km/h)"); a.set_xlabel("start hour"); a.grid(alpha=.3)
ax[0].set_ylabel("total trip time (h)"); ax[0].legend(); plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "exp2_sensitivity.png"), dpi=150); plt.close()


# ## Experiment 3: Wait vs detour crossover (one cross-region pair)
# Departure time on the x axis, time until arrival on the y axis. The best option is the lower of the two curves. The detour costs a fixed extra, while waiting costs whatever remains until the zones are safe.


pi = START if W.region[START] == "P" else int(np.where(W.region == "P")[0][0])
kc = [j for j in range(W.n) if W.region[j] == "K"]
kj = next((j for j in kc if W.name[j] == "Kota Bharu"), kc[0])
ts = np.arange(0, 1440, 10); r4w, r185, best = [], [], []
for t in ts:
    o = W.best_option(pi, kj, float(t), "full")
    r4w.append(o["r4_arrive"]-t); r185.append(o["r185_arrive"]-t); best.append(o["arrive"]-t)
r4w, r185, best = (np.array(v)/60 for v in (r4w, r185, best))
plt.figure(figsize=(9, 4)); plt.plot(ts/60, r4w, label="Route 4 (wait if needed)"); plt.plot(ts/60, r185, label="Route 185 detour")
plt.plot(ts/60, best, "k--", label="best"); plt.xlabel("departure hour"); plt.ylabel("hours until arrival"); plt.legend(); plt.grid(alpha=.3)
plt.title(f"{W.name[pi]} -> {W.name[kj]}"); plt.savefig(os.path.join(OUT_DIR, "exp3_wait_vs_detour.png"), dpi=150); plt.close()
print(f"Detour costs +{W.km(pi,kj,'R185')-W.km(pi,kj,'R4'):.0f} km vs Route 4 for this pair "
      f"(+{(W.km(pi,kj,'R185')-W.km(pi,kj,'R4'))/W.speed*60:.0f} min at {W.speed:.0f} km/h). Detour wins whenever the wait is longer than that.")


# ## Experiment 4: Grid vs brute force (justifies the data structure)
# Your real dataset only has 4 zones, so the grid will show little benefit there. We therefore also test **synthetic extra zones** (random, same hours). Both methods use the same samples, and the check below confirms they return identical hits.


def synth(m, seed=0):
    rng = np.random.default_rng(seed)
    lat = rng.uniform(4.4, 6.3, m); lon = rng.uniform(100.5, 102.6, m); x, y = to_xy(lat, lon)
    return x, y, np.full(m, 2.0), np.arange(m), np.full(m, 1140), np.full(m, 420)

rng = random.Random(1); pairs = []
Pi = [i for i in range(W.n) if W.region[i] == "P"]; Ki = [j for j in range(W.n) if W.region[j] == "K"]
for _ in range(40): pairs.append((rng.choice(Pi), rng.choice(Ki)))
samples = [sample_path(W.path(i, j, "R4"), CFG["SAMPLE_STEP_KM"]) for i, j in pairs] + \
          [sample_path(W.path(i, j, "R185"), CFG["SAMPLE_STEP_KM"]) for i, j in pairs]
pad = CFG["SAMPLE_STEP_KM"]

rows = []
for m in (4, 50, 500, 5000):
    zx, zy, zr, dz, a_s, a_e = synth(m) if m > 4 else (W.zidx.zx, W.zidx.zy, W.zidx.zr, W.zidx.dzone, W.zidx.a_s, W.zidx.a_e)
    tracemalloc.start(); t = time.perf_counter()
    idx = ZoneIndex(zx, zy, zr, dz, a_s, a_e, CFG["GRID_CELL_KM"])
    build = time.perf_counter()-t; peak = tracemalloc.get_traced_memory()[1]; tracemalloc.stop()
    t = time.perf_counter(); g = [idx.hits_grid(s, x, y, pad) for s, x, y in samples]; tg = time.perf_counter()-t
    t = time.perf_counter(); b = [idx.hits_brute(s, x, y, pad) for s, x, y in samples]; tb = time.perf_counter()-t
    same = all([(z, round(a, 6), round(c, 6)) for z, a, c in u] == [(z, round(a, 6), round(c, 6)) for z, a, c in v] for u, v in zip(g, b))
    rows.append(dict(zones=m, discs=len(zx), grid_ms=round(tg*1000, 1), brute_ms=round(tb*1000, 1), speedup=round(tb/tg, 2),
                     grid_build_ms=round(build*1000, 1), grid_peak_KB=round(peak/1024, 1), identical_results=same))
scal = pd.DataFrame(rows); display(scal); scal.to_csv(os.path.join(OUT_DIR, "exp4_grid_scaling.csv"), index=False)

print("Cell-size sweep (500 synthetic zones):")
zx, zy, zr, dz, a_s, a_e = synth(500); rows = []
for cell in (1, 2, 5, 10, 20, 50):
    tracemalloc.start(); t = time.perf_counter(); idx = ZoneIndex(zx, zy, zr, dz, a_s, a_e, cell)
    build = time.perf_counter()-t; peak = tracemalloc.get_traced_memory()[1]; tracemalloc.stop()
    t = time.perf_counter(); [idx.hits_grid(s, x, y, pad) for s, x, y in samples]; tq = time.perf_counter()-t
    rows.append(dict(cell_km=cell, occupied_cells=len(idx.grid), avg_zones_per_cell=round(np.mean([len(v) for v in idx.grid.values()]), 2),
                     build_ms=round(build*1000, 1), query_ms=round(tq*1000, 1), peak_KB=round(peak/1024, 1)))
display(pd.DataFrame(rows))


# ## Experiment 5: Does the model match Google Maps?
# Your measured numbers: Bachok to the Changkat Jering end of Perak is **344 km** via Route 4 and **410 km** via Route 185, and Gua Musang to Bachok is **193 km**. This checks how far the gateway model is off, per leg.


bach = next(((r.lat, r.lon) for r in towns.itertuples() if r.town == "Bachok"), (6.0667, 102.4))
p = GATE["CJ"]
m4 = float(r4_path_east(p, bach)[2][-1]); m5 = float(r185_path_east(p, bach)[2][-1]); gm = est(GATE["GM"], bach)
chk = pd.DataFrame([
    dict(quantity="Bachok <-> Changkat Jering via Route 4", model_km=round(m4, 1), google_km=344),
    dict(quantity="Bachok <-> Changkat Jering via Route 185", model_km=round(m5, 1), google_km=410),
    dict(quantity="Gua Musang -> Bachok", model_km=round(gm, 1), google_km=193)])
chk["error_pct"] = ((chk.model_km/chk.google_km - 1)*100).round(1); display(chk)
print("Detour extra (model):", round(m5-m4, 1), "km | Google:", 410-344, "km")


# ## Look at one tour
# Each row is a leg with departure, arrival, wait and route type.


def show(legs, t0):
    f = lambda m: f"d{int(m//1440)} {int(m%1440//60):02d}:{int(m%60):02d}"
    return pd.DataFrame([dict(frm=W.name[l["frm"]], to=W.name[l["to"]], depart=f(l["depart"]), arrive=f(l["arrive"]),
                              wait_min=round(l["wait"]), type=l["type"], route=l["route"], km=round(l["km"], 1)) for l in legs])
legsA = run_time_aware_nn(W, START, CFG["START_HOUR"]*60, "full")
display(show(legsA, CFG["START_HOUR"]*60))
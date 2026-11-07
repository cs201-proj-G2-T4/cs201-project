"""
# Approach B: Time-Expanded Graph + Dijkstra
CS201 G2T4: Time-aware touring around elephant crossing zones (Perak and Kelantan)

Inputs:
- 01_data_prep/towns.csv, 01_data_prep/distances_km.csv
- data/zones(augmented data)/: zones.csv, zone_points.csv, routes.csv, gateways.csv
- simulator.py: every tour is scored by simulate()

Builds a graph whose nodes are (place, time-of-day slot) pairs.

|     Edge      |             From → To             |  Cost   |                  Exists when               |
|---------------|-----------------------------------|---------|--------------------------------------------|
|      wait     |     (place, s) → (place, s+1)     | 1 slot  |                     always                 |
|  local drive  |         (i, s) → (j, s+k)         | k slots |      only if is safe when started at s     |
| highway drive | (gateway a, s) → (gateway b, s+k) | k slots | only if highway is safe when started at s  |

Unsafe drives do not exist in the graph, so every path Dijkstra finds is safe and waiting is chosen automatically
(another edge). Zone hours repeat daily, slots wrap around midnight and the graph has (places x slots_per_day) nodes whatever the trip length.

- Tour (go somewhere and come back): from the current (town, time), Dijkstra stops at the first unvisited town it reaches, i.e. the town that can be reached soonest and safely. Go there and repeat (time-aware Nearest Neighbour), then return to the start.
- trip time: minutes since 00:00 on day 0 of the trip. Keeps counting past 1440 (1807 = day 1, 06:07).
    Variables ending in _min, and depart / arrive / t_in / t_out, are trip times.
- time of day: minutes since midnight, 0..1439 (1140 = 19:00). Used for zone hours (active_from, active_to).
- daily window active hours are half-open [active_from, active_to): 19:00 counts as night, 07:00 as day 
    (same rule as the simulator's zone_active).
- slot: one slot_min-minute step of the day, numbered 0 .. num_daily_slots - 1 (144 slots of 10 min).
- node(place, slot): a plain tuple, compared and hashed as such by Dijkstra.
- place: a number: 0 .. n_towns-1 are the towns (towns.csv order), n_towns and above are junctions.
- km on a leg: distance driven from the start of that leg (0 at its start, leg_km at its end).
- leg: between two places
- route: between states
- point: (lat, lon, km) where km is the road distance from the start of the leg / route.
"""

# 1. Settings
#region
import math, time, os, sys, heapq, tracemalloc, itertools
from typing import NamedTuple, TypedDict
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                  # save charts to files instead of opening windows
import matplotlib.pyplot as plt
import copy, datetime

# every run gets its own folder: 04_ApproachB/outputs/run_<date>_<time>/ (CSVs, PNGs, run_log.txt)
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs",
                       "run_" + datetime.datetime.now().strftime("%Y-%m-%d_%H%M"))
os.makedirs(OUT_DIR, exist_ok=True)

class _Tee:
    """Everything printed also goes to run_log.txt"""
    def __init__(self, *streams): self.streams = streams
    def write(self, data):
        for st in self.streams: st.write(data)
    def flush(self):
        for st in self.streams: st.flush()
_log = open(os.path.join(OUT_DIR, "run_log.txt"), "w", encoding="utf-8")
sys.stdout = _Tee(sys.__stdout__, _log)
print("Run folder:", OUT_DIR)

def save_fig(name):
    """Save the current chart to OUT_DIR and close it."""
    plt.savefig(os.path.join(OUT_DIR, name), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"(saved {name})")

def save_csv(df, name):
    """Save a results table to OUT_DIR."""
    df.to_csv(os.path.join(OUT_DIR, name), index=False)
    print(f"(saved {name})")

def display(x):
    print(x.to_string() if hasattr(x, "to_string") else x)

ROOT = ".." if os.path.exists("../01_data_prep") else "."
PREP_DIR = os.path.join(ROOT, "01_data_prep")
DATA_DIR = os.path.join(ROOT, "data", "zones(augmented data)")
sys.path.insert(0, ROOT)

import simulator as sim

print(sim.load(PREP_DIR, DATA_DIR))
print()

MIN_PER_DAY   = 1440
SPEED_KMH     = sim.SPEED_KMH          # from simulator
DETOUR_FACTOR = sim.DETOUR_FACTOR      # from simulator
SLOT_MIN      = 10                     # default length of one time slot, in minutes

print(f"speed {SPEED_KMH} km/h, straight-line factor {DETOUR_FACTOR}, slot {SLOT_MIN} min")
print()
#endregion

# 2. Load the data
#region
towns = pd.read_csv(os.path.join(PREP_DIR, "towns.csv"))
dist_km = pd.read_csv(os.path.join(PREP_DIR, "distances_km.csv"), index_col="id").values

routes = pd.read_csv(os.path.join(DATA_DIR, "routes.csv")) # route_id,name,gateway_a,gateway_b,length_km,has_zones,source
gateways = pd.read_csv(os.path.join(DATA_DIR, "gateways.csv")) # name,lat,lon,source
zones = [dict(z) for z in sim.zones]
# list of
# {
#   "id": "Z1", 
#   "name": ..., 
#   "buffer_km": 1.0, 
#   "active": ("19:00", "07:00"),
#   "active_min": (1140, 420), 
#   "points": [(lat, lon, route_km), ...]
# }

ROUTE_IDS = list(routes.route_id)

print()
print(f"{len(towns)} towns, {len(zones)} zones, {len(routes)} highways")
print(f"\n{len(routes)} highways:")
display(routes[["route_id", "name", "gateway_a", "gateway_b", "length_km", "source"]])
print()
print(f"\n{len(gateways)} gateways:")
display(gateways)
print()

#endregion


# 3. The map: places and legs
#region
class ZoneSpan(NamedTuple):
    """The stretch of one leg that lies inside one zone"""
    zone_id: str
    enter_km: float      # km from the start of the leg where the car enters the zone
    exit_km: float       # km from the start of the leg where the car leaves the zone
    active_from: int     # time of day the zone becomes dangerous (1140 = 19:00)
    active_to: int       # time of day it stops being dangerous (420 = 07:00)

class Leg(TypedDict):
    """One drive between two places, as stored in Network.legs[from_place]."""
    to: int                  # destination place number
    road: str                # "local", or a route_id from routes.csv ("R4", "R185", ...)
    km: float                # length of the drive
    spans: list[ZoneSpan]    # zones it passes ([] = never near a zone, always safe)

ZONE_NEARBY_MARGIN_KM = 1.0  # quick check in zone_spans: skip a zone that is further than buffer + this from the leg

def zone_spans(geom, L, zones, step_km=0.05):
    """For a leg drawn as polyline geom [(lat, lon, km_from_start)] of length L km: the zones it passes, as
    (zone id, enter km, exit km, active_from, active_to). Zones far from the leg are skipped quickly.

    Walks the leg every step_km and keeps the spots within buffer_km of a zone
    Widens the stretch by one step on each side to be safe.
    Returns a list of ZoneSpan, one per zone the leg passes (assumes each zone is passed in one go)."""
    line = dict(points=geom) # for km_to_zone to measure distance (needs a "points" list)
    spans = [] # list of tuples [id, end of the zone (dist from the start of the road), start of the zone (dist from the start of the road), active time start, active time end]
    for zone in zones:
        closest_km = min(sim.km_to_zone(lat, lon, line) for lat, lon, _ in zone["points"])
        if closest_km > zone["buffer_km"] + ZONE_NEARBY_MARGIN_KM:
            continue # zone is nowhere near this leg
        inside = [km for km in np.arange(0, L + 1e-9, step_km) # +1e-9 so leg_km itself is sampled
                  if sim.km_to_zone(*sim.position(geom, km), zone) <= zone["buffer_km"]]
        if inside:
            spans.append(ZoneSpan(zone["id"], max(0.0, inside[0] - step_km), min(L, inside[-1] + step_km), *zone["active_min"]))
    return spans

class Network:
    """The road map: every place and every leg between places. Contains where each leg passes a zone.
    
    Attributes
        places: list of {"name", "state", "latitude", "longitude"}
        index(dict): place name -> place number
        name(list): place number -> place name
        n_towns: places 0 .. n_towns-1 are towns, the rest are junctions
        legs(dict): place number -> list of Leg leaving that place
        zones_on_route(dict): route_id -> list of zones that lie on that highway ([] for a zone-free road)
    """
    def __init__(self, towns, dist_km, routes, gateways, zones, speed=SPEED_KMH):
        self.speed, self.n_towns = speed, len(towns)
        places = towns[["name", "state", "latitude", "longitude"]].to_dict("records")
        idx = {p["name"]: i for i, p in enumerate(places)}
        for g in gateways.itertuples():                         # gateways that are not towns -> junctions
            if g.name not in idx:
                nearest_town = min(range(self.n_towns), key=lambda i: sim.haversine_km(g.lat, g.lon, places[i]["latitude"],
                                                                           places[i]["longitude"]))
                idx[g.name] = len(places)
                places.append(dict(name=g.name, state=places[nearest_town]["state"], latitude=g.lat, longitude=g.lon))
        self.places, self.index, self.name = places, idx, [p["name"] for p in places]
        coords = lambda i: (places[i]["latitude"], places[i]["longitude"])
        self.legs = {i: [] for i in range(len(places))}        # one empty list(to, road, km, spans) per place (towns AND junctions)

        # local legs (same state)
        for i in range(len(places)):
            for j in range(len(places)):
                if j == i or places[j]["state"] != places[i]["state"]:
                    continue
                are_both_towns = i < self.n_towns and j < self.n_towns
                km = float(dist_km[i, j]) if are_both_towns else sim.haversine_km(*coords(i), *coords(j)) * DETOUR_FACTOR
                spans = zone_spans([(*coords(i), 0.0), (*coords(j), km)], km, zones)
                self.legs[i].append(Leg(to=j, road="local", km=km, spans=spans))

        zones_on_route = {r.route_id: [] for r in routes.itertuples()}
        for zone in zones:
            if zone.get("route") in zones_on_route:
                zones_on_route[zone["route"]].append(zone)
        self.zones_on_route = zones_on_route
            
        for r in routes.itertuples():
            L, gateway_a, gateway_b = float(r.length_km), idx[r.gateway_a], idx[r.gateway_b]
            pts = sorted((p for z in zones_on_route[r.route_id] for p in z["points"]), key=lambda p: p[2])
            geom = [(*coords(gateway_a), 0.0)] + pts + [(*coords(gateway_b), L)] # start gateway, all the points (of danger zones on this highway/route), end gateway
            spans = zone_spans(geom, L, zones)

            # the spans when driving the other way (b → a)
            reverse = [ZoneSpan(zid, L - span_end, L - span_start, active_from, active_to) 
                        for zid, span_start, span_end, active_from, active_to in spans]

            self.legs[gateway_a].append(Leg(to=gateway_b, road=r.route_id, km=L, spans=spans))
            self.legs[gateway_b].append(Leg(to=gateway_a, road=r.route_id, km=L, spans=reverse))

    def minutes(self, km):
        """Driving time in minutes."""
        return km / self.speed * 60

    def link(self, from_place, to_place, road):
        """The Leg from one place to another along a given road ("local" or a route_id)."""
        return next(l for l in self.legs[from_place] if l["to"] == to_place and l["road"] == road)

t = time.perf_counter()
net = Network(towns, dist_km, routes, gateways, zones)

print()
print(f"{len(net.places)} places ({net.n_towns} towns + {len(net.places) - net.n_towns} junction), network built in "
      f"{time.perf_counter() - t:.1f} s")
print()

# list of legs that has danger zones
# sanity check for the report: which legs pass a zone (shows same-state legs can, too)
legs_through_zones = [(net.name[a], net.name[leg["to"]], leg["road"], [s.zone_id for s in leg["spans"]])
                      for a, legs in net.legs.items() for leg in legs if leg["spans"]]
print(f"{len(legs_through_zones)} of {sum(map(len, net.legs.values()))} legs pass a zone, for example:")
display(pd.DataFrame(legs_through_zones[:10], columns=["from", "to", "road", "zones"]))
print()
#endregion

# 4. Is a leg safe if I start it at time t?
#region
def in_window(target, start, end):
    """Is this time of day inside the active hours?"""
    if start < end:
        return start <= target < end
    return target >= start or target < end

def overlaps_daily_window(t_in, t_out, active_from, active_to):
    """Does minutes since start of day 0 [t_in, t_out] overlap the daily window [active_from, active_to) (minutes since midnight)?"""
    if in_window(t_in % MIN_PER_DAY, active_from, active_to):
        return True
    minutes_until_it_starts = (active_from - t_in) % MIN_PER_DAY
    return t_out - t_in > minutes_until_it_starts

def leg_safe(net, leg, depart_min):
    for span in leg["spans"]:
        t_in = depart_min + net.minutes(span.enter_km)
        t_out = depart_min + net.minutes(span.exit_km)
        if overlaps_daily_window(t_in, t_out, span.active_from, span.active_to):
            return False
    return True
#endregion

# 5. The data structure: a time-expanded graph, stored compactly
#region
class StaticEdge(NamedTuple):
    """A leg that touches no zone: can be driven from any slot."""
    to: int              # destination place number
    slots: int           # driving time rounded UP to whole slots (at least 1)
    minutes: float       # exact driving time
    road: str            # "local" or route_id

class TimedEdge(NamedTuple):
    """A leg that touches a zone: can only be driven from slots where safe_at_slot is True."""
    to: int
    slots: int
    minutes: float
    road: str
    safe_at_slot: list[bool]   # safe_at_slot[s] = is it safe to start this leg at slot s?

class TimeExpandedGraph:
    def __init__(self, net, slot_min=SLOT_MIN):
        self.net, self.slot = net, slot_min
        self.num_daily_slots = MIN_PER_DAY // slot_min

        # function to calculate how many slots that drive takes
        num_slots = lambda km: max(1, math.ceil(net.minutes(km) / slot_min))   # round up to whole slots

        self.static = {place: [StaticEdge(leg["to"], num_slots(leg["km"]), net.minutes(leg["km"]), leg["road"])
                           for leg in legs if not leg["spans"]] for place, legs in net.legs.items()}
        self.timed = {place: [TimedEdge(leg["to"], num_slots(leg["km"]), net.minutes(leg["km"]), leg["road"],
                           [leg_safe(net, leg, s * slot_min) for s in range(self.num_daily_slots)]) # array of T/F if its safe to start at each slot
                          for leg in legs if leg["spans"]] for place, legs in net.legs.items()}

    def edges(self, node):
        """Every move from node = (place, slot).
        Yields (next_node, cost_in_slots, exact_minutes, road). Slots wrap past midnight."""
        place, slot = node
        n = self.num_daily_slots
        yield (place, (slot + 1) % n), 1, 0.0, "wait"
        for e in self.static[place]:
            yield (e.to, (slot + e.slots) % n), e.slots, e.minutes, e.road
        for e in self.timed[place]:
            if e.safe_at_slot[slot]: yield (e.to, (slot + e.slots) % n), e.slots, e.minutes, e.road

    @property
    def n_nodes(self):
        """Number of (place, slot) nodes."""
        return len(self.net.places) * self.num_daily_slots
    @property
    def n_edges(self):
        """Number of edges a full adjacency list would hold (each wait and drive, at every slot it exists)."""
        per_slot = sum(1 + len(v) for v in self.static.values())
        return per_slot * self.num_daily_slots + sum(sum(e.safe_at_slot) for v in self.timed.values() for e in v)

# summary of build
t = time.perf_counter()
graph = TimeExpandedGraph(net)
build_s = time.perf_counter() - t

print()
print(f"Time-expanded graph: {graph.n_nodes:,} nodes, {graph.n_edges:,} edges "
      f"({len(net.places)} places x {graph.num_daily_slots} slots of {graph.slot} min), built in {build_s:.2f} s, "
      f"{sum(map(len, graph.timed.values()))} legs pass a zone")
print()
#endregion

# 6. Dijkstra
#region
def dijkstra(graph, start, targets):
    """Soonest safe arrival at ANY place in targets.
    Returns (end_node, steps) with steps = [(from_node, to_node, road), ...], or None if unreachable."""
    best_cost = {start: (0, 0.0)}
    came_from = {start: None} 
    heap = [((0, 0.0), start)]
    while heap:
        cost, node = heapq.heappop(heap)
        if cost > best_cost[node]:
            continue
        if node[0] in targets:
            steps, cur = [], node
            while came_from[cur] is not None:
                prev, road = came_from[cur]
                steps.append((prev, cur, road))
                cur = prev
            return node, steps[::-1]
        for next_node, slots, mins, road in graph.edges(node):
            new_cost = (cost[0] + slots, cost[1] + mins)
            if new_cost < best_cost.get(next_node, (math.inf, 0.0)):
                best_cost[next_node], came_from[next_node] = new_cost, (node, road)
                heapq.heappush(heap, (new_cost, next_node))
    return None
#endregion

# 7. Building the tour
#region
def run_B(net, start, t0_min, slot_min=SLOT_MIN, graph=None, visit=None):
    """Time-aware Nearest Neighbour on the time-expanded graph. Returns the legs of the tour."""
    graph = graph or TimeExpandedGraph(net, slot_min)
    unvisited = set(visit if visit is not None else range(net.n_towns)) - {start}
    t, current_place, legs, step = t0_min, start, [], 0

    def go_to_nearest(targets):
        nonlocal t, current_place, step
        clock = math.ceil(t / graph.slot - 1e-9) # first slot at or after the real time
        res = dijkstra(graph, (current_place, clock % graph.num_daily_slots), targets)
        if res is None:
            raise RuntimeError("some towns cannot be reached safely")
        end, steps = res
        for (from_place, from_slot), (to_place, to_slot), road in steps:
            if road == "wait":
                clock += 1
                continue
            link = net.link(from_place, to_place, road)
            planned = clock * graph.slot
            depart = t if leg_safe(net, link, t) else planned
            
            arrive = depart + net.minutes(link["km"])
            legs.append(dict(step=step, frm=from_place, to=to_place, road=road, km=link["km"], depart=depart,
                             wait=depart - t, arrive=arrive))
            t = arrive
            clock += (to_slot - from_slot) % graph.num_daily_slots or graph.num_daily_slots
        current_place, step = end[0], step + 1

    while unvisited:
        go_to_nearest(unvisited)
        unvisited.discard(current_place)

    go_to_nearest({start})
    return legs
#endregion


# 8. Scoring with simulator.
#region
SIM_START = "Ipoh"
SIM_START_HOURS   = (6, 8, 18)

def to_simulator_legs(tour, net):
    """legs in sim between towns only, B includes junctions"""
    groups, current = [], []
    for leg in tour:
        current.append(leg)
        if leg["to"] < net.n_towns:
            highways = [leg["road"] for leg in current if leg["road"] != "local"]
            groups.append((current[0]["frm"], leg["to"], highways[0] if highways else "local", current))
            current = []
    return groups

def score(label, tour, t0_min, net=net):
    t, unsafe, km, num_waits = t0_min, 0, 0.0, 0
    highway_uses = {route_id: 0 for route_id in ROUTE_IDS}

    for frm, to, road, B_legs in to_simulator_legs(tour, net):
        if road == "local":
            depart = B_legs[0]["depart"]
        else:
            highway = next(leg for leg in B_legs if leg["road"] != "local")
            depart = highway["depart"] - sum(net.minutes(leg["km"]) for leg in B_legs[:B_legs.index(highway)])
            highway_uses[road] += 1

        depart = max(depart, t)
        num_waits += depart - t > 1
        arrive, leg_km, hits = sim.check_leg(frm, to, depart) if road == "local" else sim.check_leg(frm, to, depart, road)

        unsafe += bool(hits)
        km += leg_km
        t = arrive

    return dict(method=label, total_h=round((t - t0_min) / 60, 2), waits=num_waits, **highway_uses,
                total_km=round(km, 1), unsafe_legs=unsafe)

def score_baseline(start, t0_min):
    result = sim.simulate(sim.nearest_neighbour(start), t0_min)
    highway_uses = {route_id: 0 for route_id in ROUTE_IDS}
    for leg in result["legs"]:
        if leg["route"] != "local":
            highway_uses[leg["route"]] = highway_uses.get(leg["route"], 0) + 1
    return dict(method="baseline", total_h=round(result["total_min"] / 60, 2), waits=0,
                **highway_uses, total_km=round(result["total_km"], 1), unsafe_legs=result["unsafe_legs"])

def clock(trip_min):
    """Trip minutes"""
    return f"d{int(trip_min // MIN_PER_DAY)} {int(trip_min % MIN_PER_DAY // 60):02d}:{int(trip_min % 60):02d}"

def show(tour, net=net):
    """B's legs"""
    return pd.DataFrame([dict(frm=net.name[leg["frm"]], to=net.name[leg["to"]], road=leg["road"],
                              km=round(leg["km"], 1), depart=clock(leg["depart"]), wait_min=round(leg["wait"]),
                              arrive=clock(leg["arrive"])) for leg in tour])

print("\n---- Simulator run ----")
START = net.index[SIM_START]

for h in SIM_START_HOURS:
    demo_t0 = h * 60
    tour = run_B(net, START, demo_t0, graph=graph)
    print(f"Tour from {SIM_START} at {clock(demo_t0)[3:]} visits {net.n_towns} towns in {len(tour)} legs. "
        f"Highway legs and waits:")
    display(show([leg for leg in tour if leg["road"] != "local" or leg["wait"] > 1]))
    print()
    display(pd.DataFrame([score_baseline(START, demo_t0), score("Approach B", tour, demo_t0)]))
    print("\n")
#endregion


# 9. Experiments
#region
def edit_spans(n, change):
    n2 = copy.deepcopy(n)
    for legs in n2.legs.values():
        for leg in legs:
            leg["spans"] = [s2 for s in leg["spans"] if (s2 := change(s)) is not None]
    return n2

def run_on(n, start_hour, start=SIM_START):
    t0 = start_hour * 60
    tour = run_B(n, n.index[start], t0, graph=TimeExpandedGraph(n, SLOT_MIN))
    return tour, (tour[-1]["arrive"] - t0) / 60

SCENARIO = ("Kota Bharu", 6)

# ---- Experiment 1: try diff start times and locations ----
EXP1_EVERY_NTH     = 20 # start from every nth town 
EXP1_HOURS         = (0, 6, 12, 18)
STARTS = sorted(set(list(range(0, len(towns), EXP1_EVERY_NTH)) + [net.index[SCENARIO[0]], net.index[SIM_START]]))
rows = []
for st in STARTS:
    for h in EXP1_HOURS:
        t0 = h * 60
        base = score_baseline(st, t0)
        b1 = score("B", run_B(net, st, t0, graph=graph), t0)
        rows.append(dict(start=net.name[st], hour=f"{h:02d}:00",
                         baseline_h=base["total_h"], baseline_unsafe_legs=base["unsafe_legs"],
                         B_h=b1["total_h"], B_waits=b1["waits"], **{f"B_{rid}": b1[rid] for rid in ROUTE_IDS},
                         B_unsafe_legs=b1["unsafe_legs"]))
exp1 = pd.DataFrame(rows)
unsafe = exp1.baseline_unsafe_legs > 0

print("\n---- Experiment 1: try diff start times and locations ----")
print(f"Baseline drives through an active zone in {unsafe.sum()} of {len(exp1)} tours ({unsafe.mean():.0%}).")
print(f"Approach B is unsafe in {(exp1.B_unsafe_legs > 0).sum()} tours.")
print(f"B is faster than the baseline in {(exp1.B_h < exp1.baseline_h - 0.01).sum()} tours, "
      f"slower in {(exp1.B_h > exp1.baseline_h + 0.01).sum()}.")
print(f"Average: baseline {exp1.baseline_h.mean():.2f} h, B {exp1.B_h.mean():.2f} h.")
print()
display(exp1)
print("\n")

plt.figure(figsize=(5.5, 5))
plt.scatter(exp1.baseline_h[~unsafe], exp1.B_h[~unsafe], label="baseline safe", s=22)
plt.scatter(exp1.baseline_h[unsafe], exp1.B_h[unsafe], label="baseline unsafe", s=22, marker="x")
lo, hi = exp1[["baseline_h", "B_h"]].min().min() - 1, exp1[["baseline_h", "B_h"]].max().max() + 1
plt.plot([lo, hi], [lo, hi], "k--", lw=1, label="same time")

plt.xlabel("baseline total time (h)")
plt.ylabel("Approach B total time (h)")
plt.legend()

plt.grid(alpha=.3)
plt.title(f"{len(exp1)} tours, scored by simulator.py")

save_fig("exp1_baseline_vs_B.png")

# ---- Experiment 2: show diff arrival time for each departure time ----
EXP2_TRIP = ("Ipoh", "Kota Bharu")

def choice(path):
    highways = [road for _, _, road in path if road not in ("local", "wait")]
    waited = any(road == "wait" for _, _, road in path)
    if not highways:
        return "local roads only"
    return f"wait, then {highways[0]}" if waited else f"{highways[0]}"

def slots_taken(path, graph):
    n = graph.num_daily_slots
    return sum(1 if road == "wait" else ((b[1] - a[1]) % n or n) for a, b, road in path)

def shade_active_hours(zones):
    """Grey bands on an hour axis for every distinct daily window the zones use."""
    for i, (start, end) in enumerate(sorted({z["active_min"] for z in zones})):
        bands = [(start, end)] if start < end else [(start, MIN_PER_DAY), (0, end)]
        for j, (s, e) in enumerate(bands):
            plt.axvspan(s / 60, e / 60, color="grey", alpha=.12, label="zones active" if i == j == 0 else None)

def departure_sweep(graph, frm, to):
    rows = []
    for s in range(graph.num_daily_slots):
        _, path = dijkstra(graph, (frm, s), {to})
        rows.append(dict(depart_h=s * graph.slot / 60,
                        hours=slots_taken(path, graph) * graph.slot / 60,
                        choice=choice(path)))
    return pd.DataFrame(rows)

frm, to = net.index[EXP2_TRIP[0]], net.index[EXP2_TRIP[1]]
exp2 = departure_sweep(graph, frm, to)

print("\n---- Experiment 2: show diff arrival time for each departure time ----")
display(exp2.groupby("choice").agg(departures=("depart_h", "size"), avg_hours=("hours", "mean")).round(2))
print("\n")

plt.figure(figsize=(9, 3.8))
for label, mark in zip(exp2.choice.unique(), itertools.cycle("os^Dv")):
    d = exp2[exp2.choice == label]
    plt.scatter(d.depart_h, d.hours, s=14, marker=mark, label=label)
shade_active_hours(zones)

plt.xlabel(f"departure hour from {EXP2_TRIP[0]}")
plt.ylabel(f"hours until {EXP2_TRIP[1]}")
plt.xticks(range(0, 25, 2))

plt.title(f"{EXP2_TRIP[0]} -> {EXP2_TRIP[1]}: B's choice by departure time")
plt.legend()
plt.grid(alpha=.3)
save_fig("exp2_departure_sweep.png")

# ---- Experiment 2b: what if the detour road had no zones? ----
DETOUR_ROUTE = "R185"
detour_zones = {z["id"] for z in net.zones_on_route[DETOUR_ROUTE]}       # {"Z6", "Z7"}, read from the data

no_detour_zone_net = edit_spans(net, lambda s: None if s.zone_id in detour_zones else s)
exp2_detour = departure_sweep(TimeExpandedGraph(no_detour_zone_net), frm, to)
save_table(exp2_detour, "exp2b_no_detour_zones.csv")

print(f"\n---- Experiment 2b: {EXP2_TRIP[0]} -> {EXP2_TRIP[1]} without {sorted(detour_zones)} ----")
display(exp2_detour.groupby("choice").agg(departures=("depart_h", "size"), avg_hours=("hours", "mean")).round(2))

plt.figure(figsize=(9, 3.8))
plt.plot(exp2.depart_h, exp2.hours, "-", color="grey", label="real zones")
for label, mark in zip(exp2_detour.choice.unique(), itertools.cycle("os^Dv")):
    d = exp2_detour[exp2_detour.choice == label]
    plt.scatter(d.depart_h, d.hours, s=14, marker=mark, label=f"no {DETOUR_ROUTE} zones: {label}")
shade_active_hours(zones)
plt.xlabel(f"departure hour from {EXP2_TRIP[0]}"); plt.ylabel(f"hours until {EXP2_TRIP[1]}"); plt.xticks(range(0, 25, 2))
plt.title(f"With vs without zones on {DETOUR_ROUTE}"); plt.legend(); plt.grid(alpha=.3); save_fig("exp2b_no_detour_zones.png")

# ---- Experiment 3a: slot size -> accuracy vs speed ----
# Smaller slots round travel times less (closer to the exact clock) but give more nodes and edges to search.
EXP3_SLOT_SIZES = (60, 30, 15, 10, 5)

start, t0 = net.index[SCENARIO[0]], SCENARIO[1] * 60
rows = []
for slot in EXP3_SLOT_SIZES:
    timer = time.perf_counter()
    g3 = TimeExpandedGraph(net, slot)
    build = time.perf_counter() - timer

    timer = time.perf_counter()
    b3 = run_B(net, start, t0, graph=g3)
    runtime = time.perf_counter() - timer

    s = score("B", b3, t0)
    rows.append(dict(slot_min=slot, nodes=g3.n_nodes, edges=g3.n_edges, build_s=round(build, 2),
                     tour_s=round(runtime, 2), total_h=s["total_h"], unsafe_legs=s["unsafe_legs"]))

print("\n---- Experiment 3a: slot size ----")
exp3a = pd.DataFrame(rows)
display(exp3a)
print("\n")

fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))

ax[0].plot(exp3a.slot_min, exp3a.total_h, "o-")
ax[0].set_ylabel("total trip time (h)")

ax[0].set_title("Rounding cost (scored by simulator)")

ax[1].plot(exp3a.slot_min, exp3a.tour_s, "o-")
ax[1].set_ylabel("tour runtime (s)")

ax[1].set_title("Search cost")
for a in ax:
    a.set_xlabel("slot size (min)")
    a.grid(alpha=.3)

plt.tight_layout()
save_fig("exp3a_slot_size.png")


# ---- Experiment 3b: storage -> compact vs full adjacency list ----
# Same graph, two ways to store it. Compact keeps each leg once (+ a safe/unsafe table for zone legs),
# the full list writes out every edge at every slot. Dijkstra sees the same edges either way.
EXP3B_FULL_MIN_SLOT = 15     # full list not built below this slot size (too much memory)

def full_adjacency(graph):
    return {(place, s): list(graph.edges((place, s))) for place in graph.static for s in range(graph.num_daily_slots)}

def peak_mb(build):
    """Peak memory (MB) while running build(), and what build() returned."""
    tracemalloc.start()
    result = build()
    peak = tracemalloc.get_traced_memory()[1] / 2**20
    tracemalloc.stop()
    return peak, result

rows = []
for slot in EXP3_SLOT_SIZES:
    compact_MB, g3b = peak_mb(lambda: TimeExpandedGraph(net, slot))   # one build, measured and kept
    full_MB = peak_mb(lambda: full_adjacency(g3b))[0] if slot >= EXP3B_FULL_MIN_SLOT else None

    rows.append(dict(slot_min=slot, edges=g3b.n_edges,
                     compact_MB=round(compact_MB, 1),
                     full_list_MB=round(full_MB, 1) if full_MB is not None else None))

print("\n---- Experiment 3b: compact vs full adjacency list ----")
exp3b = pd.DataFrame(rows)
display(exp3b)
print("\n")

plt.figure(figsize=(5.5, 3.8))
plt.plot(exp3b.slot_min, exp3b.compact_MB, "o-", label="compact")
full = exp3b.dropna(subset=["full_list_MB"])
plt.plot(full.slot_min, full.full_list_MB, "s--", label="full adjacency list")

plt.xlabel("slot size (min)")
plt.ylabel("peak memory (MB)")

plt.title("Memory: same graph, two storages")
plt.legend()
plt.grid(alpha=.3)
save_fig("exp3b_memory.png")

# ---- Experiment 4: scaling with the number of places ----
# Build the Network on n towns.
# Complexity: V = P*S nodes, compact memory O(P^2 + T*S), full adjacency list O(P^2 * S).
# Towns are sampled at random, in proportion per state. The gateway towns and the start town are always kept.
EXP4_TOWN_COUNTS = (20, 40, 80, 120, len(towns))   # towns in the network
EXP4_SEEDS       = (1, 2, 3)                       # random samples per size (one is enough for the full set)

class CountingGraph(TimeExpandedGraph):
    """Same graph. Counts how many nodes Dijkstra settles (it calls edges() once per settled node)."""
    settled = 0
    def edges(self, node):
        self.settled += 1
        return super().edges(node)

def sample_towns(n, seed):
    """n town ids: the must-keep towns, plus a random sample from each state in proportion to its size."""
    keep = {net.index[name] for name in list(gateways.name) + [SCENARIO[0]] if net.index[name] < len(towns)}
    if n >= len(towns):
        return sorted(towns.id)
    rng, picked = np.random.default_rng(seed), set(keep)
    for state, group in towns.groupby("state"):
        want = round(n * len(group) / len(towns)) - len(keep & set(group.id))
        pool = sorted(set(group.id) - keep)
        picked |= set(rng.choice(pool, size=max(0, want), replace=False).tolist())
    return sorted(picked)

rows = []
for n_towns in EXP4_TOWN_COUNTS:
    for seed in (EXP4_SEEDS if n_towns < len(towns) else EXP4_SEEDS[:1]):
        ids = sample_towns(n_towns, seed)
        sub_towns = towns.iloc[ids].reset_index(drop=True)
        sub_dist = dist_km[np.ix_(ids, ids)]

        timer = time.perf_counter()
        n4 = Network(sub_towns, sub_dist, routes, gateways, zones)
        net_s = time.perf_counter() - timer

        g4 = CountingGraph(n4)                                   # default slot size, used for the tour
        timer = time.perf_counter()
        tour4 = run_B(n4, n4.index[SCENARIO[0]], SCENARIO[1] * 60, graph=g4)
        tour_s = time.perf_counter() - timer

        rows.append(dict(towns=len(ids), seed=seed, places=len(n4.places), nodes=g4.n_nodes, edges=g4.n_edges,
                         timed_legs=sum(map(len, g4.timed.values())), network_s=round(net_s, 2),
                         nodes_settled=g4.settled, tour_s=round(tour_s, 2),
                         B_own_h=round((tour4[-1]["arrive"] - SCENARIO[1] * 60) / 60, 2)))

exp4 = pd.DataFrame(rows)
exp4_med = exp4.groupby("places").median(numeric_only=True).reset_index()   # median over the random samples

print("\n----- Experiment 4: scaling with the number of places ----")
display(exp4)
display(exp4_med[["places", "nodes", "edges", "timed_legs", "nodes_settled", "tour_s"]])

S = graph.num_daily_slots
same_state_share = (towns.state.value_counts() ** 2).sum() / len(towns) ** 2
c_theory = same_state_share * S

# Plots: growth against P, with a P^2 reference scaled to the largest size
P = exp4_med.places
P_ref_line = np.linspace(0, P.max(), 200)
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))

ax[0].plot(P, exp4_med.edges, "o-", label="edges")
ax[0].plot(P_ref_line, c_theory * P_ref_line ** 2, "k:", label=f"predicted ≈ {c_theory:.0f}·P²")
ax[0].set_title("Graph size")
ax[0].set_ylabel("edges")

ax[1].plot(P, exp4_med.nodes_settled, "o-")
ax[1].set_title("Dijkstra work for one tour")
ax[1].set_ylabel("nodes settled")

for a in ax:
    a.set_xlabel("places in the network (P)")
    a.grid(alpha=.3)
ax[0].legend()

plt.tight_layout()
save_fig("exp4_scaling.png")
print("\n")

# ---- Experiment 5: how much of B is plain Nearest Neighbour ----
tour, _ = run_on(net, SIM_START_HOURS[-1])
towns_in_order = [tour[0]["frm"]] + [r["to"] for r in tour if r["to"] < net.n_towns]
visited, same = {towns_in_order[0]}, 0

for cur, nxt in zip(towns_in_order, towns_in_order[1:-1]):     # skip the final return home
    local = [l for l in net.legs[cur] if l["road"] == "local" and l["to"] not in visited]
    same += bool(local) and min(local, key=lambda l: l["km"])["to"] == nxt
    visited.add(nxt)
steps = len(towns_in_order) - 2
print("\n--- Experiment 5: how much of B is plain Nearest Neighbour --------------------------------")
print(f"\n{same} of {steps} steps are plain NN")
print (f"{steps - same} differ (state crossings or zone-driven)\n")

# ---- Experiment 6: if no zones, compared to simulator ----
no_zone_net = edit_spans(net, lambda s: None)
start = net.index[SIM_START]
rows = []
for h in SIM_START_HOURS:
    t_free, _ = run_on(no_zone_net, h)
    t_safe, _ = run_on(net, h)
    base, free, safe = (score_baseline(start, h * 60), score("B no zones", t_free, h * 60),
                        score("B", t_safe, h * 60))
    rows.append(dict(start=f"{h:02d}:00", baseline_h=base["total_h"], baseline_unsafe=base["unsafe_legs"],
                     B_no_zones_h=free["total_h"], B_no_zones_unsafe=free["unsafe_legs"],
                     B_h=safe["total_h"], B_unsafe=safe["unsafe_legs"]))
print("\n----- Experiment 6: if no zones, compared to simulator ----")
exp6 = pd.DataFrame(rows)
display(exp6)
save_csv(exp6, "exp6_no_zones.csv")

# ---- Experiment 7: change the danger hours ----
WINDOWS = [(None, None, "real (mixed)"), (19, 7, "all 19-07"), (7, 19, "all 07-19 (day)")]
rows = []
for frm, to, label in WINDOWS:
    n_w = net if frm is None else edit_spans(net, lambda s: s._replace(active_from=frm * 60, active_to=to * 60))
    for h in SIM_START_HOURS:
        t, hrs = run_on(n_w, h)

        unsafe=sum(not leg_safe(n_w, n_w.link(r["frm"], r["to"], r["road"]), r["depart"]) for r in t)

        rows.append(dict(windows=label, start=f"{h:02d}:00", B_own_h=round(hrs, 2), highways=[r["road"] for r in t if r["road"] != "local"],
                         waits=sum(r["wait"] > 1 for r in t), unsafe=unsafe))
print("\n---- Experiment 7: change the danger hours ----")
exp7 = pd.DataFrame(rows)
display(exp7)
save_csv(exp7, "exp7_danger_hours.csv")
#endregion

print("\nAll outputs saved to", OUT_DIR)

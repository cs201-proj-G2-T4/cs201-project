"""Compare Approach A and Approach B on the same start towns and start hours (CS201 G2T4)

Run from the repo root:  python 05_comparison/compare_A_B.py   (takes about 7 min)
Saves:
  outputs/compare_A_B.csv          one row per approach, start town and start hour
  outputs/compare_A_B_summary.csv  averages for each approach + how long the first tour takes

How it works:
- A and B both start running their experiments as soon as the file is loaded, so we can't just
  import them. Instead we only run the top part of each file (settings, data, classes, functions):
    A: everything before "# ---- Experiment 1"
    B: everything before the "---- Simulator run ----" section
  We point __file__ at a temp folder so this doesn't make new run folders in A's or B's outputs.
- Both approaches' tours are checked with simulator.py, so the hours, km and unsafe legs are fair to compare.
- Time and memory are measured in separate runs, because measuring memory slows the code down a lot.
    First tour: A builds its World and then works out each leg's path and zones while it goes
    (so the first tour is slow). B builds all the legs and the whole graph before it starts.
    Later tours: A reuses what it saved, B reuses its graph.
"""
import os, sys, time, tracemalloc, tempfile, statistics, datetime
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
OUT_DIR = os.path.join(ROOT, "05_comparison", "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

START_TOWNS = ("Ipoh", "Kota Bharu", "Gerik", "Kampung Jeli")
START_HOURS = (0, 6, 8, 12, 18)
TIME_REPEATS = 3          # time each tour 3 times and use the middle one

A_FILE = os.path.join("03_ApproachA", "approach_A_time_aware_nn.py")
B_FILE = os.path.join("04_ApproachB", "approach_B_time_expanded.py")
A_CUT = "# ---- Experiment 1"
B_CUT = 'print("\\n---- Simulator run ----")'


def load_definitions(path, cut, argv):
    """Run the top of the file, up to the line that starts with cut, and return its variables."""
    with open(path, encoding="utf-8") as f:
        src = f.read()
    pos = src.find("\n" + cut)
    if pos < 0:
        sys.exit(f"ERROR: could not find '{cut}' in {path}. Was the file restructured?")
    tmp = tempfile.mkdtemp(prefix="cmp_")
    ns = {"__name__": "loaded_for_comparison", "__file__": os.path.join(tmp, os.path.basename(path))}
    old_argv, old_stdout = sys.argv, sys.stdout
    sys.argv = [path] + argv
    try:
        exec(compile(src[:pos], path, "exec"), ns)
    finally:
        sys.argv = old_argv
        if sys.stdout is not old_stdout:          # A and B send print() to their run_log too, so switch it back
            sys.stdout.flush()
            sys.stdout = old_stdout
            if "_log" in ns:
                ns["_log"].close()
    return ns


print("Loading Approach A (setup + functions only) ...")
A = load_definitions(A_FILE, A_CUT, ["01_data_prep", os.path.join("data", "zones(augmented data)")])
print("Loading Approach B (setup + functions only) ...")
B = load_definitions(B_FILE, B_CUT, [])
sim = B["sim"]        # simulator.py, already loaded by B (start Ipoh, 60 km/h)
assert A["W"].name == [t["name"] for t in sim.towns], "town order differs between Approach A and simulator.py"


# ---- A and B as simple functions ----
def a_build():
    return A["World"](A["towns"], A["zones_df"])

def a_tour(world, start, t0):
    return A["run_time_aware_nn"](world, start, t0, "full")

def b_build():
    net = B["Network"](B["towns"], B["dist_km"], B["routes"], B["gateways"], B["zones"])
    return net, B["TimeExpandedGraph"](net)

def b_tour(built, start, t0):
    net, graph = built
    return B["run_B"](net, start, t0, graph=graph)


# ---- score both with simulator.py ----
def score_A(legs, start, t0):
    tour = [start] + [l["to"] for l in legs]
    waits = {l["frm"]: l["wait"] for l in legs if l["wait"] > 1e-9}
    routes = {k: l["route"] for k, l in enumerate(legs) if l["route"] == "R185"}
    r = sim.simulate(tour, t0, waits, routes)
    return dict(total_h=r["total_min"] / 60, wait_h=r["wait_min"] / 60, total_km=r["total_km"],
                unsafe_legs=r["unsafe_legs"], R4=sum(l["route"] == "R4" for l in r["legs"]),
                R185=sum(l["route"] == "R185" for l in r["legs"]))

def score_B(legs, start, t0):
    s = B["score"]("B", legs, t0)
    wait_h = sum(l["wait"] for l in legs) / 60
    return dict(total_h=s["total_h"], wait_h=wait_h, total_km=s["total_km"], unsafe_legs=s["unsafe_legs"],
                R4=s.get("R4", 0), R185=s.get("R185", 0))


# ---- measuring ----
def timed(f, *args):
    t = time.perf_counter(); out = f(*args); return out, time.perf_counter() - t

def peak_mb(f, *args):
    tracemalloc.start(); out = f(*args); peak = tracemalloc.get_traced_memory()[1]; tracemalloc.stop()
    return out, peak / 2**20

# 1) First tour: how long until the first tour is done, and how much memory that takes.
#    A: build the World (fast), then the first tour works out and saves each leg's path and zones.
#    B: build all the legs and the graph first, then the first tour.
#    Memory is measured in a separate run because it slows things down.
REF_TOWN, REF_HOUR = "Ipoh", 8
ref = A["W"].name.index(REF_TOWN)
print(f"Measuring one-time cost (first tour from {REF_TOWN} {REF_HOUR:02d}:00) ...")
a_world, a_build_s = timed(a_build)
_, a_first_s = timed(a_tour, a_world, ref, REF_HOUR * 60)       # a_world has everything saved now, we reuse it below
_, a_first_mb = peak_mb(lambda: a_tour(a_build(), ref, REF_HOUR * 60))
b_built, b_build_s = timed(b_build)
_, b_first_s = timed(b_tour, b_built, ref, REF_HOUR * 60)
_, b_first_mb = peak_mb(lambda: b_tour(b_build(), ref, REF_HOUR * 60))
cost = pd.DataFrame([
    dict(approach="A: time-aware NN + grid index", build_s=round(a_build_s, 2), first_tour_s=round(a_first_s, 2),
         time_to_first_tour_s=round(a_build_s + a_first_s, 2), peak_MB_to_first_tour=round(a_first_mb, 1),
         note="lazy: leg paths and zone hits are computed and cached during the first tour"),
    dict(approach="B: time-expanded graph + Dijkstra", build_s=round(b_build_s, 2), first_tour_s=round(b_first_s, 2),
         time_to_first_tour_s=round(b_build_s + b_first_s, 2), peak_MB_to_first_tour=round(b_first_mb, 1),
         note="eager: every leg's zone spans and the slot table are built before any tour")])

# 2) Every tour after that, same start towns and hours for both (A reuses its World, B its graph)
rows = []
for town in START_TOWNS:
    start = A["W"].name.index(town)
    assert B["net"].name[start] == town
    for h in START_HOURS:
        t0 = h * 60
        a_times, b_times = [], []
        for _ in range(TIME_REPEATS):
            legs_a, s = timed(a_tour, a_world, start, t0); a_times.append(s)
            legs_b, s = timed(b_tour, b_built, start, t0); b_times.append(s)
        _, a_mb = peak_mb(a_tour, a_world, start, t0)
        _, b_mb = peak_mb(b_tour, b_built, start, t0)
        for name, legs, scorer, tour_s, tour_mb in (
                ("A: time-aware NN + grid index", legs_a, score_A, statistics.median(a_times), a_mb),
                ("B: time-expanded graph + Dijkstra", legs_b, score_B, statistics.median(b_times), b_mb)):
            sc = scorer(legs, start, t0)
            rows.append(dict(approach=name, start_town=town, start_hour=f"{h:02d}:00",
                             total_h=round(sc["total_h"], 2), wait_h=round(sc["wait_h"], 2),
                             total_km=round(sc["total_km"], 1), R4_legs=sc["R4"], R185_legs=sc["R185"],
                             unsafe_legs=sc["unsafe_legs"],
                             tour_ms=round(tour_s * 1000, 1), tour_peak_MB=round(tour_mb, 2)))
        print(f"  {town:13s} {h:02d}:00  A {rows[-2]['total_h']:6.2f} h  B {rows[-1]['total_h']:6.2f} h", flush=True)

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT_DIR, "compare_A_B.csv"), index=False)

# compare A and B per scenario, then average
wide = df.pivot_table(index=["start_town", "start_hour"], columns="approach", values="total_h")
diff = (wide.iloc[:, 1] - wide.iloc[:, 0])       # B minus A, in hours
summary = df.groupby("approach").agg(
    scenarios=("total_h", "size"), avg_total_h=("total_h", "mean"), avg_wait_h=("wait_h", "mean"),
    unsafe_legs=("unsafe_legs", "sum"), median_later_tour_ms=("tour_ms", "median"),
    max_later_tour_peak_MB=("tour_peak_MB", "max")).round(2).reset_index()
summary = summary.merge(cost, on="approach")
summary.to_csv(os.path.join(OUT_DIR, "compare_A_B_summary.csv"), index=False)

pd.set_option("display.width", 200)
print("\n" + summary.to_string(index=False))
print(f"\n{len(wide)} scenarios. B minus A total time: same (|diff| < 0.05 h) in {(diff.abs() < 0.05).sum()}, "
      f"B faster in {(diff <= -0.05).sum()}, A faster in {(diff >= 0.05).sum()}; "
      f"largest gap {diff.abs().max():.2f} h.")
print("Run at", datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), "| saved to", OUT_DIR)

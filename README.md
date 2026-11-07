# CS201 Project G2T4: Touring Perak and Kelantan around elephant crossings

Classic TSP assumes every road is always safe to drive. That isn't true on the roads between Perak and Kelantan, where elephants cross at night. Our TSP variant visits 161 towns in Perak and Kelantan but never drives through an elephant zone while it is active. To avoid a zone, the driver can wait in a town or take the Route 185 detour instead of Route 4.

We built two approaches and compare them with a classic Nearest Neighbour baseline:

- **Approach A:** time-aware Nearest Neighbour, using a grid over the zones to find which zones a road passes.
- **Approach B:** a time-expanded graph (each node is a place at a 10-minute time slot) searched with Dijkstra.

## Folder structure

```
cs201-project/
├── simulator.py                  shared simulator + baseline. Every tour (baseline, A, B) is checked with this
├── data/
│   ├── raw/                      the GeoNames files we were given (MY.txt, Metadata.txt, Read Me.txt, xlsx copy)
│   └── zones(augmented data)/    data we added ourselves (see "Data" below)
├── 01_data_prep/                 picks the towns from MY.txt and works out the distances
├── 02_baseline_simulator/        classic Nearest Neighbour baseline and its results (output/)
├── 03_ApproachA/                 Approach A script and its results (outputs/run_<date>_<time>/)
├── 04_ApproachB/                 Approach B script and its results (outputs/run_<date>_<time>/)
└── 05_comparison/                runs A and B on the same starts and compares them (outputs/)
```

## What you need

- Python 3.9 or newer
- numpy, pandas, matplotlib
- Jupyter (only for the two notebooks)

```
pip install numpy pandas matplotlib jupyter
```

No other libraries are used.

## How to run

Run everything from the repo root (the `cs201-project` folder). The steps are in order, but the outputs of steps 1 and 2 are already in the repo, so you can skip straight to step 3 if you want.

| Step | What | Command | Time |
|---|---|---|---|
| 1 | Data prep: makes `towns.csv`, `distances_km.csv` and `summary.txt` in `01_data_prep/` | open `01_data_prep/01_data_prep.ipynb` and Run All | < 1 min |
| 2 | Baseline: classic NN, checked with the simulator for every start hour | open `02_baseline_simulator/02_baseline_simulator.ipynb` and Run All | < 1 min |
| 3 | Approach A | `python 03_ApproachA/approach_A_time_aware_nn.py "01_data_prep" "data/zones(augmented data)"` | ~2 min |
| 4 | Approach B | `python 04_ApproachB/approach_B_time_expanded.py` | ~2.5 min |
| 5 | A vs B comparison | `python 05_comparison/compare_A_B.py` | ~7 min |

Notes:
- Keep the quotes around `"data/zones(augmented data)"`, because the folder name has a space in it.
- On Windows, use `py` instead of `python` if `python` isn't found.
- A and B make a new folder `outputs/run_<date>_<time>/` every time they run. It has the CSVs, charts and a `run_log.txt` with everything the script printed. Old runs are not overwritten.
- Step 5 overwrites the two CSVs in `05_comparison/outputs/`. The tour results will be the same, but the runtime and memory numbers depend on the computer.
- You only need `data/zones(augmented data)/make_synthetic_zones.py` if you change the zones or the length of Route 185. It rewrites `zones.csv`, `zone_points.csv` and `zone_sources.csv`. Run it with `python "data/zones(augmented data)/make_synthetic_zones.py"`.

## Results used in our slides

| Slide content | File |
|---|---|
| Baseline drives through active zones | `02_baseline_simulator/output/` |
| Approach A experiments | `03_ApproachA/outputs/run_2026-11-07_1657/` |
| Approach B experiments | `04_ApproachB/outputs/run_2026-11-07_1659/` |
| A vs B (trip time, runtime, memory) | `05_comparison/outputs/compare_A_B_summary.csv` and `compare_A_B.csv` |

## Data

**From the given dataset (MY.txt)**
- Columns used: geonameid, name / asciiname, latitude, longitude, feature class, feature code, admin1 (state), admin2 (district) and population.
- Records kept: populated places (feature class P) in Perak (admin1 07) and Kelantan (admin1 03).
- We dropped feature codes that aren't real towns now: PPLX (part of a city), PPLQ (abandoned), PPLH (historical) and PPLW (destroyed). That leaves 3,494 places.
- From those we picked 161 towns (88 in Perak, 73 in Kelantan):
  - every town that has a population (101 towns)
  - then more towns until each district has at least 7, picking each new town as far from the others in its district as possible (60 towns)
  - places less than 1 km from a town we already picked are skipped as duplicates
- Why 161: it's big enough that every district, including the rural areas where elephants cross, has towns, and small enough that every experiment runs in a few minutes.
- Road distance = straight-line (haversine) distance x 1.3, since the dataset has no road distances. `01_data_prep/summary.txt` lists a few sample distances so they can be checked against Google Maps.

**Data we added (`data/zones(augmented data)/`)**
- `zones.csv` and `zone_points.csv`: 12 elephant crossing zones, each with its location, the hours it is active and a buffer distance.
  - Z1 to Z4 are real crossing spots on Route 4, taken from news reports.
  - Z5 to Z12 are synthetic zones, which we added with course approval. They are marked `evidence_type=synthetic`.
- `zone_sources.csv`: the news source for each real zone, and the nearest forest reserve in MY.txt for each synthetic zone.
- `routes.csv`: the two highways between the states.
  - Route 4: Gerik to Kampung Jeli, 126 km (Google Maps).
  - Route 185: Simpang Pulai to Gua Musang, 156 km.
- `gateways.csv`: where each highway starts and ends. All four are places in MY.txt.
  - Simpang Pulai is tagged as Pahang (06) in GeoNames, even though it is in Kinta, Perak. So we use it as a junction, not as one of our towns.

## AI use

We used AI tools in this project. See the AI acknowledgement section of our slides for what was used, where, and the example prompts.

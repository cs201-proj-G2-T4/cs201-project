"""
Adds our 8 made-up (synthetic) elephant zones, Z5-Z12, to the 4 zones from the news, Z1-Z4.

Run from the repo root:   python "data/zones(augmented data)/make_synthetic_zones.py"
It rewrites zones.csv, zone_points.csv and zone_sources.csv. Z1-Z4 are kept as they are.

Why: we could only find news reports for 4 zones, so (with course approval) we added 8 more.
They're all marked evidence_type=synthetic in zones.csv. We put each one:
  - next to a forest reserve from MY.txt (the closest one is saved in zone_sources.csv)
  - on a road our tours actually use, so it can change a tour
  - away from all our towns, so no town is inside a zone
  - across 4 kinds of road: Route 4 (Kelantan side), Route 185, Perak local roads, Kelantan local roads

Shape: same as Z1-Z4, 9 points 0.5 km apart, so each zone is 4 km of road.
  - Route 4 / Route 185 zones sit on the highway. road_km is km from the Perak end
    (Gerik for Route 4, Simpang Pulai for Route 185).
  - Local zones: local legs are straight lines between two towns (straight line x 1.3),
    so the points sit on that line, near the place in the zone's name.
    road_km is km along that leg from the first town.
"""
import csv, math, os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
DETOUR = 1.3          # same factor as 01_data_prep
STEP_ROAD_KM = 0.5
HALF_LEN_ROAD_KM = 2.0
R = 6371.0088

def hav(a, b):
    p1, p2 = math.radians(a[0]), math.radians(b[0])
    h = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(b[1] - a[1]) / 2) ** 2
    return 2 * R * math.asin(math.sqrt(h))

def lerp(a, b, f):
    return (a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1]))

def towns():
    with open(os.path.join(ROOT, "01_data_prep", "towns.csv"), encoding="utf-8") as f:
        return {r["name"]: (float(r["latitude"]), float(r["longitude"])) for r in csv.DictReader(f)}

T = towns()
Z4_END = (5.60206, 101.59249)                  # last point of Z4 (KM82), from zone_points.csv
KUALA_BETIS = (4.9014, 101.7896)               # GeoNames 'Kampung Kuala Betis'
GUA_MUSANG = T["Gua Musang"]

# Route 185 zones use km from Simpang Pulai. We take the length from routes.csv and work
# backwards from Gua Musang to get the km of Kuala Betis and Hendrop.
def r185_length():
    with open(os.path.join(HERE, "routes.csv"), encoding="utf-8") as f:
        return next(float(r["length_km"]) for r in csv.DictReader(f) if r["route_id"] == "R185")

KUALA_BETIS_KM = r185_length() - hav(KUALA_BETIS, GUA_MUSANG) * DETOUR
HENDROP_KM = KUALA_BETIS_KM - hav(T["Hendrop"], KUALA_BETIS) * DETOUR

# For each zone: the straight bit of road A->B it sits on, the road km at A,
# road km / straight line km on that bit, and how far along A->B the zone's centre is.
ZONES = [
    # id, name, route, A, B, road_km_at_A, scale, centre_frac, buffer, start, end, rationale
    ("Z5", "Route 4 KM92 (Jeli side, Kelantan)", "R4", Z4_END, T["Kampong Delong"], 82.0, 1.37, 0.53, 1.0, "19:00", "07:00",
     "Continues the Belum-Temengor forest corridor east of Z4; same night window as Z1-Z4"),
    ("Z6", "Route 185 east of Hendrop", "R185", T["Hendrop"], KUALA_BETIS, HENDROP_KM, 1.3, 0.18, 1.0, "18:30", "06:30",
     "Highland forest stretch between Hendrop and Kuala Betis; makes the Route 185 detour not automatically safe"),
    ("Z7", "Route 185 near Kuala Betis", "R185", KUALA_BETIS, GUA_MUSANG, KUALA_BETIS_KM, 1.3, 0.25, 1.0, "20:00", "08:00",
     "Forest edge west of Gua Musang; window shifted later to model early-morning crossings"),
    ("Z8", "Gerik-Lenggong road near Kenering", "LOCAL", T["Lenggong"], T["Gerik"], 0.0, DETOUR, 0.52, 1.0, "18:00", "06:00",
     "Route 76 past Tasik Kenering / Belum Baharu, beside the same Perak forest block as Route 4"),
    ("Z9", "Gerik-Pengkalan Hulu road", "LOCAL", T["Gerik"], T["Keruh"], 0.0, DETOUR, 0.30, 0.5, "21:00", "05:00",
     "Forested road north of Gerik toward the Thai border; narrow late-night window, tight buffer"),
    ("Z10", "Route 8 near Limau Kasturi", "LOCAL", GUA_MUSANG, T["Kampong Serasa"], 0.0, DETOUR, 0.66, 1.5, "17:00", "23:00",
     "Plantation-forest edge on the Gua Musang-Kuala Krai road; dusk peak (elephants raid crops at dusk)"),
    ("Z11", "Jeli-Kuala Balah road", "LOCAL", T["Kampung Rangka Baru"], T["Kampung Bukit Jering"], 0.0, DETOUR, 0.50, 1.0, "19:00", "07:00",
     "Forested road south of Jeli, between Rangka Baru and Bukit Jering"),
    ("Z12", "Gua Musang-Pulat road", "LOCAL", GUA_MUSANG, T["Kampung Wias"], 0.0, DETOUR, 0.55, 1.0, "04:00", "09:00",
     "Interior road near Kampung Pulat; dawn-only window so a night-only rule is not enough"),
]

def build():
    zones, points, info = [], [], []
    for zid, name, route, A, B, km_a, scale, frac, buf, s, e, why in ZONES:
        seg_geo = hav(A, B)
        c_road = km_a + frac * seg_geo * scale
        pts = []
        for k in range(9):
            rk = c_road - HALF_LEN_ROAD_KM + k * STEP_ROAD_KM
            f = (rk - km_a) / scale / seg_geo
            pts.append((k, round(rk, 1), *[round(v, 5) for v in lerp(A, B, f)]))
        c = pts[4]
        zones.append(dict(zone_id=zid, name=name, road_km=round(c_road), km_start=pts[0][1], km_end=pts[-1][1],
                          center_lat=c[2], center_lon=c[3], buffer_km=buf, active_start=s, active_end=e,
                          evidence="SYNTHETIC: " + why, route=route, evidence_type="synthetic"))
        points += [dict(zone_id=zid, seq=p[0], road_km=p[1], lat=p[2], lon=p[3]) for p in pts]
        info.append((zid, (c[2], c[3]), why))
    return zones, points, info

def nearest_forest(centre):
    best = (1e9, "")
    with open(os.path.join(ROOT, "data", "raw", "MY.txt"), encoding="utf-8") as f:
        for line in f:
            c = line.rstrip("\n").split("\t")
            if c[7] in ("FRST", "RESF", "RESN", "PRK") and c[10] in ("03", "07"):
                d = hav(centre, (float(c[4]), float(c[5])))
                if d < best[0]: best = (d, c[1])
    return best

def read(name):
    with open(os.path.join(HERE, name), encoding="utf-8") as f:
        return list(csv.DictReader(f))

def write(name, rows, cols):
    with open(os.path.join(HERE, name), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)

if __name__ == "__main__":
    new_z, new_p, info = build()
    keep = lambda rows: [r for r in rows if r["zone_id"] in ("Z1", "Z2", "Z3", "Z4", "ALL")]
    old_z = keep(read("zones.csv"))
    for r in old_z: r.update(route="R4", evidence_type="news")
    zcols = ["zone_id","name","road_km","km_start","km_end","center_lat","center_lon","buffer_km",
             "active_start","active_end","evidence","route","evidence_type"]
    write("zones.csv", old_z + new_z, zcols)
    write("zone_points.csv", keep(read("zone_points.csv")) + new_p, ["zone_id","seq","road_km","lat","lon"])
    src = keep(read("zone_sources.csv"))
    for zid, c, why in info:
        d, fr = nearest_forest(c)
        src.append(dict(zone_id=zid, what_it_supports=f"SYNTHETIC zone (course-approved). {why}. Nearest forest reserve in GeoNames: {fr} ({d:.1f} km)",
                        source="Synthetic, placed by G2T4 using GeoNames MY.txt", url=""))
    write("zone_sources.csv", src, ["zone_id","what_it_supports","source","url"])
    print("wrote", len(old_z) + len(new_z), "zones,", len(new_p), "new points")

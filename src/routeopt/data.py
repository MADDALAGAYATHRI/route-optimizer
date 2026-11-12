"""Synthetic shipment / route dataset.

Design goals (see report, Section 2):
  * Routes are built leg-by-leg from a hub table, so cost, time and risk all
    follow from the same physical description (no independent random numbers
    that could produce nonsensical routes).
  * Every lane contains genuine trade-offs: a short route through a high-risk
    chokepoint, a safe-but-long detour, a fast-but-expensive air option, and
    hybrid sea-air / rail options in between.
  * ~30% of shipments receive a planted *dominated decoy* (worse on every
    criterion than another option) so the evaluation can check that the
    system never recommends a strictly inferior route.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------
# Hub table: congestion (0-1), geopolitical risk (0-1), data confidence.
# kind: "port", "air", "rail", "chokepoint" (waypoint: no handling / dwell)
# confidence: how well-established the delay statistics are (higher = better)
# --------------------------------------------------------------------------
HUBS = {
    # Asia
    "Shanghai":            dict(kind="port", congestion=0.55, geo=0.15, confidence=60),
    "Shanghai PVG":        dict(kind="air",  congestion=0.35, geo=0.15, confidence=60),
    "Shenzhen":            dict(kind="port", congestion=0.45, geo=0.15, confidence=60),
    "Shenzhen SZX":        dict(kind="air",  congestion=0.30, geo=0.15, confidence=60),
    "Busan":               dict(kind="port", congestion=0.35, geo=0.10, confidence=60),
    "Busan ICN":           dict(kind="air",  congestion=0.25, geo=0.10, confidence=60),
    "Singapore":           dict(kind="port", congestion=0.40, geo=0.05, confidence=60),
    "Colombo":             dict(kind="port", congestion=0.35, geo=0.20, confidence=40),
    "Nhava Sheva":         dict(kind="port", congestion=0.60, geo=0.10, confidence=40),
    "Mumbai BOM":          dict(kind="air",  congestion=0.40, geo=0.10, confidence=40),
    "Chennai":             dict(kind="port", congestion=0.50, geo=0.10, confidence=40),
    "Chennai MAA":         dict(kind="air",  congestion=0.35, geo=0.10, confidence=40),
    # Middle East / chokepoints
    "Jebel Ali":           dict(kind="port", congestion=0.30, geo=0.35, confidence=50),
    "Dubai DXB":           dict(kind="air",  congestion=0.25, geo=0.35, confidence=50),
    "Suez Canal":          dict(kind="chokepoint", congestion=0.40, geo=0.75, confidence=25),
    "Cape of Good Hope":   dict(kind="chokepoint", congestion=0.10, geo=0.10, confidence=40),
    "Panama Canal":        dict(kind="chokepoint", congestion=0.50, geo=0.20, confidence=35),
    # Overland rail (Eurasian land bridge)
    "Khorgos":             dict(kind="rail", congestion=0.40, geo=0.45, confidence=15),
    "Malaszewicze":        dict(kind="rail", congestion=0.55, geo=0.60, confidence=15),
    # Europe
    "Rotterdam":           dict(kind="port", congestion=0.35, geo=0.05, confidence=60),
    "Hamburg":             dict(kind="port", congestion=0.40, geo=0.05, confidence=60),
    "Algeciras":           dict(kind="port", congestion=0.30, geo=0.05, confidence=50),
    "Tanger Med":          dict(kind="port", congestion=0.30, geo=0.15, confidence=40),
    "Frankfurt FRA":       dict(kind="air",  congestion=0.25, geo=0.05, confidence=60),
    # Americas
    "Los Angeles":         dict(kind="port", congestion=0.55, geo=0.05, confidence=60),
    "Oakland":             dict(kind="port", congestion=0.35, geo=0.05, confidence=50),
    "Los Angeles LAX":     dict(kind="air",  congestion=0.30, geo=0.05, confidence=60),
    "Anchorage ANC":       dict(kind="air",  congestion=0.15, geo=0.05, confidence=50),
    "New York":            dict(kind="port", congestion=0.45, geo=0.05, confidence=60),
    "New York JFK":        dict(kind="air",  congestion=0.35, geo=0.05, confidence=60),
    "Houston":             dict(kind="port", congestion=0.40, geo=0.05, confidence=60),
    "Houston IAH":         dict(kind="air",  congestion=0.25, geo=0.05, confidence=60),
    "Manzanillo":          dict(kind="port", congestion=0.45, geo=0.30, confidence=30),
    "Santos":              dict(kind="port", congestion=0.55, geo=0.20, confidence=40),
    "Sao Paulo GRU":       dict(kind="air",  congestion=0.30, geo=0.20, confidence=40),
}

# Per-mode economics. cost: USD per tonne-km. speed: km/day (incl. slow steaming)
MODES = {
    "sea":   dict(cost=0.0080, speed=650,  handling=8.0,  dwell=2.0, p_base=0.06, delay_mean=2.0),
    "rail":  dict(cost=0.0350, speed=650,  handling=12.0, dwell=1.5, p_base=0.08, delay_mean=1.5),
    "air":   dict(cost=0.3500, speed=6000, handling=40.0, dwell=0.5, p_base=0.03, delay_mean=0.5),
    "truck": dict(cost=0.0900, speed=550,  handling=5.0,  dwell=0.3, p_base=0.03, delay_mean=0.5),
}
CHOKEPOINT_FEE = {"Suez Canal": 4.0, "Panama Canal": 3.0}  # USD / tonne

# --------------------------------------------------------------------------
# Lanes and route templates. A template alternates node, (mode, km), node, ...
# --------------------------------------------------------------------------
LANES = {
    "Shanghai-Rotterdam": [
        ("Sea via Singapore & Suez", "short_via_chokepoint",
         ["Shanghai", ("sea", 3800), "Singapore", ("sea", 8600), "Suez Canal", ("sea", 6000), "Rotterdam"]),
        ("Sea via Singapore & Cape", "safe_detour",
         ["Shanghai", ("sea", 3800), "Singapore", ("sea", 9700), "Cape of Good Hope", ("sea", 11300), "Rotterdam"]),
        ("Rail land bridge via Khorgos", "overland",
         ["Shanghai", ("rail", 4300), "Khorgos", ("rail", 5100), "Malaszewicze", ("truck", 1200), "Rotterdam"]),
        ("Sea-air via Jebel Ali / Dubai", "hybrid",
         ["Shanghai", ("sea", 10500), "Jebel Ali", ("truck", 40), "Dubai DXB", ("air", 5200), "Frankfurt FRA", ("truck", 450), "Rotterdam"]),
        ("Air via Frankfurt", "air",
         ["Shanghai PVG", ("air", 8900), "Frankfurt FRA", ("truck", 450), "Rotterdam"]),
    ],
    "Shenzhen-Los Angeles": [
        ("Direct transpacific sea", "direct_sea",
         ["Shenzhen", ("sea", 11700), "Los Angeles"]),
        ("Sea via Busan transshipment", "transship",
         ["Shenzhen", ("sea", 2100), "Busan", ("sea", 9600), "Los Angeles"]),
        ("Sea to Oakland + truck", "alt_port",
         ["Shenzhen", ("sea", 11100), "Oakland", ("truck", 600), "Los Angeles"]),
        ("Sea-air via Busan / Incheon", "hybrid",
         ["Shenzhen", ("sea", 2100), "Busan", ("truck", 330), "Busan ICN", ("air", 9600), "Los Angeles LAX"]),
        ("Air via Anchorage", "air",
         ["Shenzhen SZX", ("air", 8400), "Anchorage ANC", ("air", 3700), "Los Angeles LAX"]),
    ],
    "Nhava Sheva-Hamburg": [
        ("Sea via Suez", "short_via_chokepoint",
         ["Nhava Sheva", ("sea", 4900), "Suez Canal", ("sea", 6600), "Hamburg"]),
        ("Sea via Cape of Good Hope", "safe_detour",
         ["Nhava Sheva", ("sea", 9100), "Cape of Good Hope", ("sea", 12000), "Hamburg"]),
        ("Sea via Jebel Ali & Suez", "transship",
         ["Nhava Sheva", ("sea", 1900), "Jebel Ali", ("sea", 3900), "Suez Canal", ("sea", 6600), "Hamburg"]),
        ("Sea-air via Dubai", "hybrid",
         ["Nhava Sheva", ("sea", 1900), "Jebel Ali", ("truck", 40), "Dubai DXB", ("air", 4800), "Frankfurt FRA", ("truck", 500), "Hamburg"]),
        ("Air via Frankfurt", "air",
         ["Mumbai BOM", ("air", 6600), "Frankfurt FRA", ("truck", 500), "Hamburg"]),
    ],
    "Chennai-New York": [
        ("Sea via Colombo & Suez", "short_via_chokepoint",
         ["Chennai", ("sea", 700), "Colombo", ("sea", 5500), "Suez Canal", ("sea", 8900), "New York"]),
        ("Sea via Colombo & Cape", "safe_detour",
         ["Chennai", ("sea", 700), "Colombo", ("sea", 8300), "Cape of Good Hope", ("sea", 12600), "New York"]),
        ("Sea via Singapore & Panama", "east_route",
         ["Chennai", ("sea", 2900), "Singapore", ("sea", 18700), "Panama Canal", ("sea", 3700), "New York"]),
        ("Air via Frankfurt", "air",
         ["Chennai MAA", ("air", 7600), "Frankfurt FRA", ("air", 6200), "New York JFK"]),
    ],
    "Santos-Rotterdam": [
        ("Direct sea", "direct_sea",
         ["Santos", ("sea", 10000), "Rotterdam"]),
        ("Sea via Algeciras transshipment", "transship",
         ["Santos", ("sea", 8000), "Algeciras", ("sea", 2600), "Rotterdam"]),
        ("Sea via Tanger Med transshipment", "transship",
         ["Santos", ("sea", 7900), "Tanger Med", ("sea", 2650), "Rotterdam"]),
        ("Air via Frankfurt", "air",
         ["Santos", ("truck", 80), "Sao Paulo GRU", ("air", 9800), "Frankfurt FRA", ("truck", 450), "Rotterdam"]),
    ],
    "Busan-Houston": [
        ("Sea via Panama", "short_via_chokepoint",
         ["Busan", ("sea", 13800), "Panama Canal", ("sea", 2900), "Houston"]),
        ("Sea to LA + rail landbridge", "intermodal",
         ["Busan", ("sea", 9600), "Los Angeles", ("rail", 2500), "Houston"]),
        ("Sea via Manzanillo + rail", "alt_port",
         ["Busan", ("sea", 10600), "Manzanillo", ("rail", 2100), "Houston"]),
        ("Air via Anchorage", "air",
         ["Busan ICN", ("air", 5600), "Anchorage ANC", ("air", 5200), "Houston IAH"]),
    ],
}

# Cargo types -> (weight range t, value USD/t, profile probabilities)
CARGO = {
    "apparel":     dict(weight=(10, 24), value=8_000,   profiles={"cost_first": .6, "balanced": .4}),
    "auto_parts":  dict(weight=(8, 22),  value=15_000,  profiles={"balanced": .5, "time_critical": .5}),
    "electronics": dict(weight=(1, 6),   value=120_000, profiles={"time_critical": .5, "balanced": .3, "risk_averse": .2}),
    "pharma":      dict(weight=(1, 5),   value=250_000, profiles={"risk_averse": .7, "time_critical": .3}),
    "perishables": dict(weight=(5, 18),  value=6_000,   profiles={"time_critical": .8, "balanced": .2}),
    "bulk_goods":  dict(weight=(15, 26), value=2_000,   profiles={"cost_first": .8, "balanced": .2}),
}


def _parse_template(template):
    nodes = template[0::2]
    edges = template[1::2]
    return nodes, edges


def build_route(route_id, name, archetype, template, ctx, rng):
    """Turn a template into a fully attributed route (legs, cost, base time, risk inputs)."""
    nodes, edges = _parse_template(template)
    w = ctx["weight_t"]
    legs, events = [], []
    cost, base_days = 0.0, 0.0

    for i, (mode, km) in enumerate(edges):
        m = MODES[mode]
        rate = m["cost"] * ctx["rate_mult"][mode]
        leg_cost = rate * km * w
        leg_days = km / m["speed"]
        legs.append(dict(origin=nodes[i], destination=nodes[i + 1], mode=mode, km=km,
                         cost_usd=round(leg_cost, 2), days=round(leg_days, 2)))
        cost += leg_cost
        base_days += leg_days
        # en-route weather / operational delay on long legs
        events.append(dict(source=f"{mode} leg {nodes[i]}->{nodes[i+1]}", p=m["p_base"] * min(km / 8000, 1.0),
                           delay_mean=m["delay_mean"], geo=0.0, confidence=50))

    hubs_used = []
    for j, node in enumerate(nodes):
        h = HUBS[node]
        cong = min(0.95, h["congestion"] * ctx["season"] * rng.uniform(0.9, 1.1))
        geo = min(0.95, h["geo"] * rng.uniform(0.9, 1.1))
        hubs_used.append(dict(name=node, kind=h["kind"], congestion=round(cong, 3), geo=round(geo, 3)))
        # mode used to handle the node = mode of adjacent leg (prefer the outgoing leg)
        mode = edges[j][0] if j < len(edges) else edges[j - 1][0]
        m = MODES[mode]
        if h["kind"] == "chokepoint":
            cost += CHOKEPOINT_FEE.get(node, 0.0) * w
            base_days += 0.5
            p = 0.05 + 0.25 * cong + 0.10 * geo
            dmean = 2.0 + 6.0 * cong
        else:
            cost += m["handling"] * w + 150.0  # fixed doc / terminal fee
            base_days += m["dwell"] * (1 + cong)
            p = m["p_base"] + 0.25 * cong + 0.10 * geo
            dmean = m["delay_mean"] + (6.0 if mode == "sea" else 2.0) * cong
        events.append(dict(source=f"{h['kind']} {node}", p=round(min(p, 0.9), 4),
                           delay_mean=round(dmean, 3), geo=round(geo, 3), confidence=h["confidence"]))

    return dict(
        route_id=route_id, name=name, archetype=archetype,
        hubs=[h["name"] for h in hubs_used],
        legs=legs, hub_detail=hubs_used, events=events,
        cost_usd=round(cost, 2),
        base_transit_days=round(base_days, 2),
        congestion_risk=round(max(h["congestion"] for h in hubs_used), 3),
        geopolitical_risk=round(max(h["geo"] for h in hubs_used), 3),
        estimate_confidence=int(min(e["confidence"] for e in events)),
        is_decoy=False,
    )


def make_decoy(route, route_id, rng):
    """A strictly dominated copy: same path + an extra congested transshipment + higher cost."""
    d = json.loads(json.dumps(route))
    d["route_id"] = route_id
    d["name"] = route["name"] + " (extra transshipment)"
    d["archetype"] = "dominated_decoy"
    d["is_decoy"] = True
    extra_cong = 0.75
    d["cost_usd"] = round(route["cost_usd"] * rng.uniform(1.08, 1.20) + 300, 2)
    d["base_transit_days"] = round(route["base_transit_days"] + rng.uniform(2.0, 4.0), 2)
    d["congestion_risk"] = max(route["congestion_risk"], extra_cong)
    d["events"].append(dict(source="port extra transshipment", p=0.30, delay_mean=4.0, geo=0.0, confidence=20))
    d["estimate_confidence"] = 20
    return d


def generate_dataset(n_shipments: int = 40, seed: int = 7) -> list[dict]:
    rng = np.random.default_rng(seed)
    lanes = list(LANES)
    cargos = list(CARGO)
    shipments = []
    for s in range(n_shipments):
        lane = lanes[s % len(lanes)]
        cargo = cargos[rng.integers(len(cargos))]
        c = CARGO[cargo]
        profile = rng.choice(list(c["profiles"]), p=list(c["profiles"].values()))
        weight = round(float(rng.uniform(*c["weight"])), 1)
        ctx = dict(
            weight_t=weight,
            season=float(rng.uniform(0.8, 1.3)),  # seasonal congestion multiplier
            rate_mult={m: float(rng.uniform(0.8, 1.2)) for m in MODES},  # market rate noise
        )
        templates = LANES[lane]
        n_opts = int(rng.integers(3, min(5, len(templates)) + 1))
        idx = sorted(rng.choice(len(templates), size=n_opts, replace=False))
        routes = [build_route(f"R{k+1}", *templates[i], ctx, rng) for k, i in enumerate(idx)]

        # planted dominated decoy in ~30% of shipments (keeps total options <= 5)
        if rng.random() < 0.30 and len(routes) < 5:
            sea_like = [r for r in routes if r["archetype"] != "air"] or routes
            base = sea_like[rng.integers(len(sea_like))]
            routes.append(make_decoy(base, f"R{len(routes)+1}", rng))

        base_days = [r["base_transit_days"] for r in routes]
        costs = [r["cost_usd"] for r in routes]
        deadline = None
        if rng.random() < {"time_critical": .9, "balanced": .6, "risk_averse": .5, "cost_first": .3}[profile]:
            # deadline relative to the cheapest ("default") route: binds sometimes,
            # tighter for time-critical cargo, never placed only to force air freight
            cheapest_days = base_days[int(np.argmin(costs))]
            lo_f, hi_f = (0.45, 1.10) if profile == "time_critical" else (0.80, 1.35)
            deadline = round(cheapest_days * rng.uniform(lo_f, hi_f) + 3, 1)
        budget, planted_conflict = None, False
        if rng.random() < 0.4:
            if deadline is None:
                budget = round(min(costs) * float(rng.uniform(1.3, 3.0)), -1)
            elif rng.random() < 0.2:
                # deliberately conflicting: budget only fits routes too slow for the deadline
                budget = round(min(costs) * float(rng.uniform(1.1, 1.4)), -1)
                planted_conflict = True
            else:
                # budget anchored to the cheapest route that plausibly meets the deadline
                fast = [c for c, d in zip(costs, base_days) if d + 5 <= deadline] or [min(costs)]
                budget = round(min(fast) * float(rng.uniform(1.1, 1.6)), -1)
        max_geo = 0.6 if (cargo == "pharma" or (profile == "risk_averse" and rng.random() < .5)) else None

        shipments.append(dict(
            shipment_id=f"SHP-{s+1:03d}",
            lane=lane,
            origin=lane.split("-")[0],
            destination=lane.split("-")[1],
            cargo_type=cargo,
            weight_t=weight,
            cargo_value_usd=round(weight * c["value"], -2),
            priority_profile=str(profile),
            constraints=dict(
                deadline_days=deadline,
                min_on_time_prob=0.85 if profile == "time_critical" else 0.75,
                budget_usd=budget,
                max_geopolitical_risk=max_geo,
            ),
            planted_conflict=planted_conflict,
            routes=routes,
        ))
    return shipments


def save_dataset(shipments, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(shipments, indent=1), encoding="utf-8")


def load_dataset(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def routes_table(shipments):
    """Flat one-row-per-route view (for CSV export / inspection)."""
    import pandas as pd
    rows = []
    for s in shipments:
        for r in s["routes"]:
            rows.append(dict(
                shipment_id=s["shipment_id"], lane=s["lane"], cargo_type=s["cargo_type"],
                profile=s["priority_profile"], route_id=r["route_id"], route=r["name"],
                archetype=r["archetype"], hubs=" > ".join(r["hubs"]),
                modes="+".join(dict.fromkeys(l["mode"] for l in r["legs"])),
                cost_usd=r["cost_usd"], base_transit_days=r["base_transit_days"],
                congestion_risk=r["congestion_risk"], geopolitical_risk=r["geopolitical_risk"],
                estimate_confidence=r["estimate_confidence"], is_decoy=r["is_decoy"],
            ))
    return pd.DataFrame(rows)

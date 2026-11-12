"""Route decision engine.

Pipeline (each stage is individually explainable to an operator):

  1. Metrics      -- cost (deterministic) + Monte Carlo time/delay metrics.
  2. Constraints  -- hard rules that must never be traded off:
                     deadline (P(on time) >= min_on_time_prob), budget cap,
                     maximum geopolitical exposure.
  3. Pareto       -- drop feasible routes that another feasible route beats on
                     every criterion. Assumption-free: holds for ANY weights.
  4. Score        -- weighted sum of penalties (lower is better) over 4 criteria:
                     cost, expected transit days, delay probability,
                     geopolitical risk. Weights come from the shipment's
                     priority profile.

Normalisation (fixed reference units, not min-max):
    penalty_i = (x_i - best_i) / unit_i
  unit_i is a fixed, operator-meaningful amount of each criterion:
    cost  -> 5% of the cargo value    (a $5k premium matters on a $40k bulk
                                       load, not on a $750k pharma load)
    time  -> 14 days | delay prob -> 20 pp | geo risk -> 0.30 on the index
  Total penalty (x100 = "penalty points", 0 = best on every criterion) is a
  weighted L1 distance to the ideal point. Because units do not depend on
  which other routes are present, adding or removing an option can never
  flip the order of two others (no rank reversal), unlike min-max scaling.
"""
from __future__ import annotations

from .simulate import route_metrics

CRITERIA = ["cost_usd", "expected_days", "delay_prob", "geopolitical_risk"]

CRITERION_LABEL = {
    "cost_usd": "cost",
    "expected_days": "expected transit time",
    "delay_prob": "delay probability",
    "geopolitical_risk": "geopolitical risk",
}

# weights: cost, time, delay, geo  (each row sums to 1)
PROFILES = {
    "balanced":      {"cost_usd": .35, "expected_days": .25, "delay_prob": .25, "geopolitical_risk": .15},
    "cost_first":    {"cost_usd": .60, "expected_days": .15, "delay_prob": .15, "geopolitical_risk": .10},
    "time_critical": {"cost_usd": .15, "expected_days": .45, "delay_prob": .30, "geopolitical_risk": .10},
    "risk_averse":   {"cost_usd": .20, "expected_days": .15, "delay_prob": .30, "geopolitical_risk": .35},
}


def units(shipment: dict) -> dict:
    """Fixed reference unit per criterion (one unit = a significant difference)."""
    return {
        "cost_usd": max(0.05 * shipment["cargo_value_usd"], 500.0),
        "expected_days": 14.0,
        "delay_prob": 0.20,
        "geopolitical_risk": 0.30,
    }


# Below this fraction of a unit, a difference is treated as immaterial in explanations.
IMMATERIAL = 0.10


# Dominance tolerance (absorbs Monte Carlo noise)
EPS = {"cost_usd": 0.0, "expected_days": 0.1, "delay_prob": 0.01, "geopolitical_risk": 0.0}


def compute_metrics(shipment: dict, salt: int = 0, bias: dict | None = None, n: int | None = None) -> dict:
    """route_id -> metrics dict (cost, MC time metrics, risk)."""
    out = {}
    for r in shipment["routes"]:
        kw = {"salt": salt, "bias": (bias or {}).get(r["route_id"], 1.0)}
        if n:
            kw["n"] = n
        m = route_metrics(shipment, r, **kw)
        m["cost_usd"] = r["cost_usd"]
        m["geopolitical_risk"] = r["geopolitical_risk"]
        out[r["route_id"]] = m
    return out


def check_constraints(shipment: dict, m: dict) -> list[str]:
    c = shipment["constraints"]
    v = []
    if c["deadline_days"] is not None and m["on_time_prob"] < c["min_on_time_prob"]:
        v.append(f"on-time probability {m['on_time_prob']:.0%} < required {c['min_on_time_prob']:.0%} "
                 f"for the {c['deadline_days']:.0f}-day deadline")
    if c["budget_usd"] is not None and m["cost_usd"] > c["budget_usd"]:
        v.append(f"cost ${m['cost_usd']:,.0f} exceeds budget ${c['budget_usd']:,.0f}")
    if c["max_geopolitical_risk"] is not None and m["geopolitical_risk"] > c["max_geopolitical_risk"]:
        v.append(f"geopolitical risk {m['geopolitical_risk']:.2f} above limit {c['max_geopolitical_risk']:.2f}")
    return v


def respects_risk_cap(shipment: dict, m: dict) -> bool:
    cap = shipment["constraints"]["max_geopolitical_risk"]
    return cap is None or m["geopolitical_risk"] <= cap


def violation_size(shipment: dict, m: dict) -> float:
    """Relative size of *relaxable* violations (deadline, budget) for the fallback.

    Constraints are tiered: the geopolitical-risk cap is a safety/compliance
    limit and is never relaxed; deadline and budget are commercial limits that
    a planner may relax on escalation.
    """
    c = shipment["constraints"]
    s = 0.0
    if c["deadline_days"] is not None:
        s += max(0.0, c["min_on_time_prob"] - m["on_time_prob"]) / c["min_on_time_prob"]
    if c["budget_usd"] is not None:
        s += max(0.0, m["cost_usd"] - c["budget_usd"]) / c["budget_usd"]
    return s


def dominates(a: dict, b: dict) -> bool:
    no_worse = all(a[k] <= b[k] + EPS[k] for k in CRITERIA)
    better = any(a[k] < b[k] - EPS[k] for k in CRITERIA)
    return no_worse and better


def score_candidates(metrics: dict, candidates: list[str], weights: dict, unit: dict) -> dict:
    """route_id -> (penalty points, normalised gaps in units, weighted contributions)."""
    out = {}
    best = {k: min(metrics[r][k] for r in candidates) for k in CRITERIA}
    for r in candidates:
        norm = {k: (metrics[r][k] - best[k]) / unit[k] for k in CRITERIA}
        contrib = {k: 100.0 * weights[k] * norm[k] for k in CRITERIA}
        out[r] = dict(score=sum(contrib.values()), norm=norm, contrib=contrib)
    return out


def decide(shipment: dict, profile: str | None = None, weights: dict | None = None,
           metrics: dict | None = None) -> dict:
    profile = profile or shipment["priority_profile"]
    weights = weights or PROFILES[profile]
    metrics = metrics or compute_metrics(shipment)
    names = {r["route_id"]: r for r in shipment["routes"]}
    unit = units(shipment)

    rows = {rid: dict(route_id=rid, name=names[rid]["name"], archetype=names[rid]["archetype"],
                      hubs=names[rid]["hubs"], is_decoy=names[rid]["is_decoy"],
                      **metrics[rid], violations=check_constraints(shipment, metrics[rid]),
                      dominated_by=None, score=None, norm=None, contrib=None)
            for rid in metrics}

    feasible = [rid for rid, r in rows.items() if not r["violations"]]
    fallback = not feasible
    if feasible:
        pool = feasible
    else:  # fallback: keep the risk cap, relax deadline/budget
        pool = [rid for rid in rows if respects_risk_cap(shipment, metrics[rid])] or list(rows)

    # Pareto filter within the pool
    candidates = []
    for rid in pool:
        dom = next((o for o in pool if o != rid and dominates(metrics[o], metrics[rid])), None)
        if dom:
            rows[rid]["dominated_by"] = dom
        else:
            candidates.append(rid)

    if fallback:
        # No route meets every hard constraint: pick the smallest violation, flag for escalation.
        ranking = sorted(candidates, key=lambda r: (violation_size(shipment, metrics[r]), metrics[r]["expected_days"]))
        scored = score_candidates(metrics, candidates, weights, unit)
        for rid in candidates:
            rows[rid].update(scored[rid])
    else:
        scored = score_candidates(metrics, candidates, weights, unit)
        for rid in candidates:
            rows[rid].update(scored[rid])
        ranking = sorted(candidates, key=lambda r: scored[r]["score"])

    return dict(
        shipment_id=shipment["shipment_id"],
        profile=profile,
        weights=weights,
        units=unit,
        constraints=shipment["constraints"],
        routes=rows,
        candidates=candidates,
        ranking=ranking,
        winner=ranking[0],
        runner_up=ranking[1] if len(ranking) > 1 else None,
        fallback=fallback,
    )

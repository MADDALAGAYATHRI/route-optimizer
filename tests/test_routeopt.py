import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from routeopt import compute_metrics, decide, explain, generate_dataset  # noqa: E402
from routeopt.scoring import dominates  # noqa: E402
from routeopt.simulate import route_metrics, simulate_route  # noqa: E402


# ---------------------------------------------------------------- helpers
def mk_route(rid, cost, days, p=0.1, geo=0.1, conf=50, decoy=False):
    return dict(route_id=rid, name=f"route {rid}", archetype="test", hubs=["A", "B"], legs=[],
                events=[dict(source="hub", p=p, delay_mean=3.0, geo=geo, confidence=conf)],
                cost_usd=cost, base_transit_days=days, congestion_risk=0.3, geopolitical_risk=geo,
                estimate_confidence=conf, is_decoy=decoy)


def mk_ship(routes, profile="balanced", deadline=None, budget=None, max_geo=None, value=200_000):
    return dict(shipment_id="TEST", lane="A-B", origin="A", destination="B", cargo_type="electronics",
                weight_t=5, cargo_value_usd=value, priority_profile=profile,
                constraints=dict(deadline_days=deadline, min_on_time_prob=0.8, budget_usd=budget,
                                 max_geopolitical_risk=max_geo),
                routes=routes)


@pytest.fixture(scope="module")
def dataset():
    return generate_dataset(40, 7)


@pytest.fixture(scope="module")
def decisions(dataset):
    out = []
    for s in dataset:
        m = compute_metrics(s)
        out.append((s, m, decide(s, metrics=m)))
    return out


# ---------------------------------------------------------------- data
def test_generator_is_deterministic():
    a, b = generate_dataset(10, 3), generate_dataset(10, 3)
    assert [r["cost_usd"] for s in a for r in s["routes"]] == [r["cost_usd"] for s in b for r in s["routes"]]


def test_dataset_shape(dataset):
    assert 20 <= len(dataset) <= 50
    assert all(3 <= len(s["routes"]) <= 5 for s in dataset)
    assert any(r["is_decoy"] for s in dataset for r in s["routes"])


# ---------------------------------------------------------------- simulation
def test_more_risk_means_more_delay():
    r = mk_route("R1", 1000, 20, p=0.2)
    lo = simulate_route(r, n=4000, seed=1, bias=0.5).mean()
    hi = simulate_route(r, n=4000, seed=1, bias=2.0).mean()
    assert hi > lo


def test_on_time_monotone_in_deadline():
    r = mk_route("R1", 1000, 20, p=0.3)
    probs = [route_metrics(mk_ship([r], deadline=d), r)["on_time_prob"] for d in (20, 23, 26, 30)]
    assert probs == sorted(probs)


def test_low_confidence_widens_tail():
    hi_conf = simulate_route(mk_route("R1", 1, 20, p=0.3, conf=60), n=20000, seed=2)
    lo_conf = simulate_route(mk_route("R1", 1, 20, p=0.3, conf=10), n=20000, seed=2)
    assert np.percentile(lo_conf, 97) >= np.percentile(hi_conf, 97)


# ---------------------------------------------------------------- decision logic
def test_profile_changes_the_answer():
    cheap_slow = mk_route("R1", 2_000, 40, p=0.3)
    fast_dear = mk_route("R2", 30_000, 5, p=0.05)
    s = mk_ship([cheap_slow, fast_dear])
    assert decide(s, profile="cost_first")["winner"] == "R1"
    assert decide(s, profile="time_critical")["winner"] == "R2"


def test_dominated_route_never_chosen():
    good = mk_route("R1", 2_000, 20, p=0.1, geo=0.1)
    bad = mk_route("R2", 2_500, 25, p=0.3, geo=0.1, decoy=True)
    for p in ("balanced", "cost_first", "time_critical", "risk_averse"):
        d = decide(mk_ship([good, bad]), profile=p)
        assert d["winner"] == "R1" and d["routes"]["R2"]["dominated_by"] == "R1"


def test_hard_constraints_beat_score():
    risky_cheap = mk_route("R1", 1_000, 20, p=0.1, geo=0.8)
    safe_dear = mk_route("R2", 1_500, 22, p=0.1, geo=0.1)
    d = decide(mk_ship([risky_cheap, safe_dear], profile="cost_first", max_geo=0.6))
    assert d["winner"] == "R2" and d["routes"]["R1"]["violations"]


def test_fallback_never_relaxes_risk_cap():
    # nothing fits the budget; R1 breaks the risk cap, R2 only the budget
    r1 = mk_route("R1", 900, 20, geo=0.8)
    r2 = mk_route("R2", 3_000, 20, geo=0.2)
    d = decide(mk_ship([r1, r2], budget=500, max_geo=0.6))
    assert d["fallback"] and d["winner"] == "R2"


def test_immaterial_cost_difference_does_not_decide():
    # $50 on a $200k cargo is immaterial; 10 days faster is not
    r1 = mk_route("R1", 5_000, 30)
    r2 = mk_route("R2", 5_050, 20)
    d = decide(mk_ship([r1, r2], profile="cost_first"))
    assert d["winner"] == "R2"
    assert "$50 more" not in explain(d, mk_ship([r1, r2]))


def test_explanation_mentions_price_of_constraints():
    cheap_slow = mk_route("R1", 1_000, 40)
    fast = mk_route("R2", 8_000, 10)
    s = mk_ship([cheap_slow, fast], deadline=20)
    text = explain(decide(s), s)
    assert "RECOMMENDED: R2" in text and "ruled out by the deadline" in text


# ---------------------------------------------------------------- dataset-wide properties
def test_no_recommendation_violates_constraints(decisions):
    for s, m, d in decisions:
        if not d["fallback"]:
            assert not d["routes"][d["winner"]]["violations"], s["shipment_id"]


def test_no_recommendation_is_dominated(decisions):
    for s, m, d in decisions:
        feas = [r for r in d["routes"] if not d["routes"][r]["violations"]]
        assert not any(dominates(m[o], m[d["winner"]]) for o in feas if o != d["winner"]), s["shipment_id"]


def test_decoys_never_recommended(decisions):
    for s, m, d in decisions:
        assert not d["routes"][d["winner"]]["is_decoy"], s["shipment_id"]


def test_no_rank_reversal_when_removing_a_loser(decisions):
    for s, m, d in decisions:
        if d["fallback"]:
            continue
        for drop in d["candidates"]:
            if drop == d["winner"]:
                continue
            sub = dict(s, routes=[r for r in s["routes"] if r["route_id"] != drop])
            sub_m = {k: v for k, v in m.items() if k != drop}
            assert decide(sub, metrics=sub_m)["winner"] == d["winner"], (s["shipment_id"], drop)


def test_every_shipment_gets_an_explanation(decisions):
    for s, m, d in decisions:
        text = explain(d, s)
        assert f"RECOMMENDED: {d['winner']}" in text

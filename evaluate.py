"""Evaluate the decision framework on the synthetic dataset.

Writes to results/:
  summary.json        -- all evaluation metrics
  decisions.csv       -- one row per shipment (winner, metrics, robustness)
  baselines.csv       -- policy comparison table
  explanations.txt    -- operator explanation for every shipment
  fig_tradeoff.png    -- trade-off view of an example shipment
  fig_stability.png   -- robustness of every recommendation

Checks
  E1 hard-constraint compliance        E5 baseline policies (non-circular metrics)
  E2 Pareto-optimality of winners      E6 robustness (weights / estimates / MC seed)
  E3 planted decoy rejection           E7 context sensitivity (profile sweep)
  E4 directional consistency           E8 rank-reversal test vs min-max scoring
"""
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from routeopt import CRITERIA, compute_metrics, decide, explain, load_dataset  # noqa: E402
from routeopt.explain import robustness_verdict  # noqa: E402
from routeopt.scoring import check_constraints, dominates  # noqa: E402
from routeopt.sensitivity import estimate_stability, flip_points, profile_sweep, weight_stability  # noqa: E402

OUT = ROOT / "results"


def minmax_winner(metrics, cands, weights):
    """Plain min-max weighted scoring (the common textbook approach) for comparison."""
    best = {k: min(metrics[r][k] for r in cands) for k in CRITERIA}
    worst = {k: max(metrics[r][k] for r in cands) for k in CRITERIA}
    def pen(r):
        return sum(weights[k] * ((metrics[r][k] - best[k]) / (worst[k] - best[k]) if worst[k] > best[k] else 0)
                   for k in CRITERIA)
    return min(cands, key=pen)


def main(data_path=str(ROOT / "data" / "shipments.json")):
    t0 = time.time()
    OUT.mkdir(exist_ok=True)
    S = load_dataset(data_path)
    print(f"[eval] {len(S)} shipments, {sum(len(s['routes']) for s in S)} routes from {data_path}")

    rows, texts, base_rows = [], [], []
    e1_viol, e2_dom, e3 = 0, 0, Counter()
    e4 = Counter()
    e8 = Counter()
    seed_agree = []
    cache = {}

    for i, s in enumerate(S):
        m = compute_metrics(s)
        d = decide(s, metrics=m)
        cache[s["shipment_id"]] = (m, d)
        R, w = d["routes"], d["winner"]
        feasible = [r for r in R if not R[r]["violations"]]

        # E1 constraints / E2 Pareto / E3 decoys
        if not d["fallback"] and R[w]["violations"]:
            e1_viol += 1
        if any(dominates(m[o], m[w]) for o in feasible if o != w):
            e2_dom += 1
        for r in s["routes"]:
            if r["is_decoy"]:
                e3["decoys"] += 1
                e3["decoy_chosen"] += int(r["route_id"] == w)
                e3["decoy_marked_dominated"] += int(R[r["route_id"]]["dominated_by"] is not None
                                                    or bool(R[r["route_id"]]["violations"]))

        # Sensitivity (E6)
        sens = None
        if not d["fallback"] and len(d["candidates"]) > 1:
            sens = dict(weights=weight_stability(s, m, d["weights"], w),
                        estimates=estimate_stability(s, d["weights"], w),
                        flips=flip_points(s, m, d["weights"], w))
        # MC seed stability: independent simulation runs with fresh seeds
        agree = np.mean([decide(s, metrics=compute_metrics(s, salt=1000 + k))["winner"] == w for k in range(5)])
        seed_agree.append(agree)

        # E7 profile sweep
        sweep = profile_sweep(s, m)

        # E4 directional consistency across profiles (non-circular sanity)
        if not d["fallback"]:
            c_cost, c_time = R[sweep["cost_first"]], R[sweep["time_critical"]]
            e4["n"] += 1
            e4["cost_first_not_dearer"] += int(c_cost["cost_usd"] <= c_time["cost_usd"] + 1e-6)
            e4["time_critical_not_slower"] += int(c_time["expected_days"] <= c_cost["expected_days"] + 1e-6)
            c_risk = R[sweep["risk_averse"]]
            e4["risk_averse_not_riskier"] += int(
                c_risk["geopolitical_risk"] <= R[sweep["cost_first"]]["geopolitical_risk"] + 1e-6)

        # E8 rank reversal: remove each non-winning candidate, does the winner change?
        cands = d["candidates"]
        if not d["fallback"] and len(cands) >= 3:
            mm_w = minmax_winner(m, cands, d["weights"])
            for drop in cands:
                sub = [c for c in cands if c != drop]
                if drop != w:
                    e8["ours_tests"] += 1
                    sub_ship = dict(s, routes=[r for r in s["routes"] if r["route_id"] != drop])
                    sub_m = {k: v for k, v in m.items() if k != drop}
                    e8["ours_reversals"] += int(decide(sub_ship, metrics=sub_m)["winner"] != w)
                if drop != mm_w:
                    e8["minmax_tests"] += 1
                    e8["minmax_reversals"] += int(minmax_winner(m, sub, d["weights"]) != mm_w)

        # E5 baselines -- each policy picks one route
        def pick(pool, key):
            return min(pool, key=key) if pool else None
        all_r = list(R)
        policies = {
            "Ours (profile-weighted)": w,
            "Cheapest (unconstrained)": pick(all_r, lambda r: R[r]["cost_usd"]),
            "Fastest (unconstrained)": pick(all_r, lambda r: R[r]["expected_days"]),
            "Cheapest feasible": pick(feasible, lambda r: R[r]["cost_usd"]) or w,
            "Fastest feasible": pick(feasible, lambda r: R[r]["expected_days"]) or w,
        }
        for pol, r in policies.items():
            base_rows.append(dict(policy=pol, shipment_id=s["shipment_id"], profile=d["profile"], route=r,
                                  cargo_value=s["cargo_value_usd"],
                                  cost=R[r]["cost_usd"], cost_pct_value=R[r]["cost_usd"] / s["cargo_value_usd"],
                                  days=R[r]["expected_days"], p90=R[r]["p90_days"],
                                  delay=R[r]["delay_prob"], geo=R[r]["geopolitical_risk"],
                                  violates=bool(check_constraints(s, m[r])),
                                  dominated=any(dominates(m[o], m[r]) for o in all_r if o != r),
                                  penalty=decide(s, metrics=m)["routes"][r]["score"]))

        label, _ = robustness_verdict(sens)
        rows.append(dict(
            shipment_id=s["shipment_id"], lane=s["lane"], cargo=s["cargo_type"], profile=d["profile"],
            n_routes=len(R), n_feasible=len(feasible), n_candidates=len(d["candidates"]),
            decided_by=("fallback" if d["fallback"] else "constraints/dominance" if len(d["candidates"]) == 1
                        else "trade-off score"),
            winner=w, winner_route=R[w]["name"], winner_archetype=R[w]["archetype"],
            cost_usd=R[w]["cost_usd"], expected_days=round(R[w]["expected_days"], 1),
            p90_days=round(R[w]["p90_days"], 1), delay_prob=round(R[w]["delay_prob"], 3),
            geo_risk=R[w]["geopolitical_risk"],
            weight_stability=None if not sens else sens["weights"]["stability"],
            estimate_stability=None if not sens else sens["estimates"]["stability"],
            seed_agreement=agree, verdict=label or ("n/a (single option)" if not d["fallback"] else "fallback"),
            distinct_winners_across_profiles=len(set(sweep.values())),
        ))
        texts.append(explain(d, s, sens))
        print(f"[eval] {s['shipment_id']} done ({i+1}/{len(S)}) winner={w} verdict={rows[-1]['verdict']}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "decisions.csv", index=False)
    (OUT / "explanations.txt").write_text(("\n\n" + "=" * 90 + "\n\n").join(texts), encoding="utf-8")

    B = pd.DataFrame(base_rows)
    btab = B.groupby("policy", sort=False).agg(
        mean_cost_usd=("cost", "mean"), median_cost_pct_of_value=("cost_pct_value", "median"),
        mean_days=("days", "mean"), mean_p90_days=("p90", "mean"), mean_delay_prob=("delay", "mean"),
        mean_geo_risk=("geo", "mean"), constraint_violation_rate=("violates", "mean"),
        dominated_pick_rate=("dominated", "mean"))
    btab.round(3).to_csv(OUT / "baselines.csv")

    # where ours differs from cheapest-feasible: what did we buy?
    ours = B[B.policy == "Ours (profile-weighted)"].set_index("shipment_id")
    chf = B[B.policy == "Cheapest feasible"].set_index("shipment_id")
    diff = ours.route != chf.route
    trade = dict(
        n_differ=int(diff.sum()),
        by_profile=ours[diff].profile.value_counts().to_dict(),
        median_extra_cost_usd=float((ours.cost - chf.cost)[diff].median()),
        median_extra_cost_pct_of_value=float(((ours.cost - chf.cost) / ours.cargo_value * 100)[diff].median()),
        mean_days_saved=float((chf.days - ours.days)[diff].mean()),
        mean_delay_pp_saved=float(((chf.delay - ours.delay) * 100)[diff].mean()),
        mean_geo_reduction=float((chf.geo - ours.geo)[diff].mean()),
    )

    scored = df[df.decided_by == "trade-off score"]
    summary = dict(
        n_shipments=len(S), n_routes=int(df.n_routes.sum()),
        decided_by=df.decided_by.value_counts().to_dict(),
        E1_constraint_violations_in_recommendations=e1_viol,
        E1_fallback_shipments=int((df.decided_by == "fallback").sum()),
        E2_dominated_recommendations=e2_dom,
        E3_decoys=dict(e3),
        E4_directional_consistency={k: (v / e4["n"] if k != "n" else v) for k, v in e4.items()},
        E5_baselines=btab.round(3).reset_index().to_dict(orient="records"),
        E5_vs_cheapest_feasible=trade,
        E6_robustness=dict(
            verdicts=df.verdict.value_counts().to_dict(),
            mean_weight_stability=float(scored.weight_stability.mean()),
            mean_estimate_stability=float(scored.estimate_stability.mean()),
            mc_seed_agreement=float(np.mean(seed_agree)),
        ),
        E7_context_sensitivity=dict(
            share_with_profile_dependent_winner=float((df.distinct_winners_across_profiles > 1).mean()),
            distinct_winner_counts=df.distinct_winners_across_profiles.value_counts().sort_index().to_dict()),
        E8_rank_reversal=dict(ours=f"{e8['ours_reversals']}/{e8['ours_tests']}",
                              minmax=f"{e8['minmax_reversals']}/{e8['minmax_tests']}"),
        runtime_s=round(time.time() - t0, 1),
    )
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    make_figures(S, cache, df)
    print(json.dumps({k: v for k, v in summary.items() if k != "E5_baselines"}, indent=2, default=float))
    print(btab.round(3).to_string())
    return summary


def make_figures(S, cache, df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Fig 1: trade-off view of an illustrative shipment (most candidates, decided by trade-off)
    # prefer a non-obvious decision: winner is neither the cheapest nor air freight
    pool = df[(df.decided_by == "trade-off score") & (df.winner_archetype != "air")].copy()
    pool["not_cheapest"] = [cache[sid][1]["routes"][w]["cost_usd"] >
                            min(cache[sid][1]["routes"][c]["cost_usd"] for c in cache[sid][1]["candidates"])
                            for sid, w in zip(pool.shipment_id, pool.winner)]
    pick = pool.sort_values(["not_cheapest", "n_candidates", "n_routes"], ascending=False).iloc[0].shipment_id
    s = next(x for x in S if x["shipment_id"] == pick)
    m, d = cache[pick]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    for rid, r in d["routes"].items():
        status = ("winner" if rid == d["winner"] else "infeasible" if r["violations"]
                  else "dominated" if r["dominated_by"] else "candidate")
        col = {"winner": "#1b7f3b", "candidate": "#2f6db5", "dominated": "#9a9a9a", "infeasible": "#c0392b"}[status]
        ax.errorbar(r["cost_usd"], r["expected_days"],
                    yerr=[[0.0], [r["p90_days"] - r["expected_days"]]],
                    fmt="none", ecolor=col, alpha=.6, capsize=3)
        ax.scatter(r["cost_usd"], r["expected_days"], s=(500 if status == "winner" else 60) + 700 * r["delay_prob"],
                   c=col, marker="*" if status == "winner" else "o", alpha=.55, edgecolors="k", linewidths=.6,
                   zorder=3)
        ax.annotate(f"{rid} {r['name']}\n{status}, geo {r['geopolitical_risk']:.2f}, delay {r['delay_prob']:.0%}",
                    (r["cost_usd"], r["expected_days"]), textcoords="offset points", xytext=(8, 4), fontsize=6.5)
    ax.set_xscale("log")
    from matplotlib.ticker import FuncFormatter, NullFormatter
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v/1000:,.0f}k" if v >= 1000 else f"${v:,.0f}"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    lo_c = min(r["cost_usd"] for r in d["routes"].values())
    hi_c = max(r["cost_usd"] for r in d["routes"].values())
    ax.set_xticks([t for t in [500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000]
                   if lo_c / 2 <= t <= hi_c * 2])
    ax.set_xlabel("Total cost (USD, log scale)")
    ax.set_ylabel("Expected transit (days)\nwhisker = P90")
    ax.set_title(f"{pick}: {s['cargo_type'].replace('_', ' ')}, {s['lane']}, profile={d['profile']}  "
                 f"(marker size = delay probability)", fontsize=8.5)
    ax.grid(alpha=.3)
    ax.margins(x=0.35, y=0.15)
    fig.tight_layout()
    fig.savefig(OUT / "fig_tradeoff.png", dpi=200)
    plt.close(fig)

    # Fig 2: robustness of each trade-off decision
    sc = df[df.decided_by == "trade-off score"].sort_values("estimate_stability")
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    x = np.arange(len(sc))
    ax.bar(x - .2, sc.weight_stability, .4, label="priority-weight perturbation", color="#2f6db5")
    ax.bar(x + .2, sc.estimate_stability, .4, label="risk-estimate perturbation", color="#e39b2d")
    ax.axhline(.8, ls="--", c="k", lw=.8)
    ax.axhline(.6, ls=":", c="k", lw=.8)
    ax.text(len(sc) - .5, .81, "robust", fontsize=7, ha="right")
    ax.text(len(sc) - .5, .61, "moderate", fontsize=7, ha="right")
    ax.set_xticks(x)
    ax.set_xticklabels([sid.replace("SHP-", "") for sid in sc.shipment_id], fontsize=6, rotation=90)
    ax.set_ylabel("share keeping\nsame winner")
    ax.set_xlabel("shipment (decided by trade-off score)")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "fig_stability.png", dpi=200)
    plt.close(fig)
    print(f"[eval] figures written (example shipment {pick})")


if __name__ == "__main__":
    main(*sys.argv[1:])

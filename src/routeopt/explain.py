"""Turn a decision into an explanation a logistics operator can act on.

Explanations are generated from the numbers (template-based), never invented:
every sentence maps to a metric, a constraint or a score contribution, so the
text can be audited against the table printed beside it.
"""
from __future__ import annotations

from .scoring import CRITERIA, CRITERION_LABEL, IMMATERIAL

PROFILE_LABEL = {
    "balanced": "balanced", "cost_first": "cost-first",
    "time_critical": "time-critical", "risk_averse": "risk-averse",
}

ROBUST, MODERATE = 0.80, 0.60


def _diff_phrase(k, w, o):
    """Describe winner value w vs other value o for criterion k."""
    if k == "cost_usd":
        d = w - o
        pct = d / o * 100
        return (f"costs ${d:,.0f} more (+{pct:.0f}%)" if d > 0
                else f"is ${-d:,.0f} cheaper ({pct:.0f}%)")
    if k == "expected_days":
        d = w - o
        return (f"takes {d:.1f} days longer on average ({w:.1f} vs {o:.1f})" if d > 0
                else f"arrives {-d:.1f} days sooner on average ({w:.1f} vs {o:.1f})")
    if k == "delay_prob":
        return f"has a {w:.0%} delay probability vs {o:.0%}"
    return f"has geopolitical risk {w:.2f} vs {o:.2f}"


def compare(decision, a, b):
    """Material advantages / drawbacks of route a relative to route b, ordered by score impact."""
    ra, rb = decision["routes"][a], decision["routes"][b]
    adv, drw = [], []
    for k in CRITERIA:
        wa, wb = ra[k], rb[k]
        gap = abs(wa - wb) / decision["units"][k]
        if gap < IMMATERIAL:
            continue  # immaterial difference: don't mention it
        impact = decision["weights"][k] * gap
        (adv if wa < wb else drw).append((impact, k, _diff_phrase(k, wa, wb)))
    adv.sort(reverse=True)
    drw.sort(reverse=True)
    return adv, drw


def robustness_verdict(sens):
    if not sens:
        return None, None
    ws = sens["weights"]["stability"]
    es = sens["estimates"]["stability"]
    worst = min(ws, es)
    label = "ROBUST" if worst >= ROBUST else ("MODERATE" if worst >= MODERATE else "CLOSE CALL")
    return label, (f"winner holds in {ws:.0%} of priority-weight perturbations and "
                   f"{es:.0%} of risk-estimate perturbations")


def explain(decision, shipment, sens=None) -> str:
    R = decision["routes"]
    w = decision["winner"]
    c = decision["constraints"]
    lines = []
    cons = []
    if c["deadline_days"]:
        cons.append(f"deadline {c['deadline_days']:.0f} d at >= {c['min_on_time_prob']:.0%} on-time")
    if c["budget_usd"]:
        cons.append(f"budget ${c['budget_usd']:,.0f}")
    if c["max_geopolitical_risk"] is not None:
        cons.append(f"geo-risk <= {c['max_geopolitical_risk']:.2f}")
    lines.append(f"Shipment {shipment['shipment_id']}: {shipment['cargo_type'].replace('_', ' ')}, "
                 f"{shipment['weight_t']} t, {shipment['origin']} -> {shipment['destination']}")
    lines.append(f"Priority: {PROFILE_LABEL[decision['profile']]}"
                 + (f" | Constraints: {', '.join(cons)}" if cons else " | No hard constraints"))
    lines.append("")

    rw = R[w]
    if decision["fallback"]:
        lines.append("WARNING: no route satisfies every hard constraint. Showing the least-violating option "
                     "-- escalate to a planner (relax deadline/budget or split the shipment).")
        for v in rw["violations"]:
            lines.append(f"  - still violates: {v}")
    lines.append(f"RECOMMENDED: {w} - {rw['name']}")
    lines.append(f"  ${rw['cost_usd']:,.0f} | {rw['expected_days']:.1f} days expected "
                 f"(P90 {rw['p90_days']:.1f}) | delay prob {rw['delay_prob']:.0%} | "
                 f"geo risk {rw['geopolitical_risk']:.2f}"
                 + (f" | on-time {rw['on_time_prob']:.0%}" if rw["on_time_prob"] is not None else ""))
    lines.append("")

    # price of constraints: is a cheaper route excluded by a hard constraint?
    cheapest = min(R, key=lambda k: R[k]["cost_usd"])
    if cheapest != w and R[cheapest]["violations"]:
        why = R[cheapest]["violations"][0].split(" ")[0]
        why = {"on-time": "the deadline", "cost": "the budget", "geopolitical": "the geopolitical-risk limit"}[why]
        lines.append(f"Price of constraints: the cheapest option, {cheapest} ({R[cheapest]['name']}, "
                     f"${R[cheapest]['cost_usd']:,.0f}), is ruled out by {why}; the recommendation costs "
                     f"${rw['cost_usd'] - R[cheapest]['cost_usd']:,.0f} more.")
        lines.append("")

    ru = decision["runner_up"]
    if decision["fallback"]:
        lines.append(f"Why {w}: among routes that respect the geopolitical-risk limit (never relaxed), it has the "
                     f"smallest relative breach of the relaxable constraints (deadline, budget).")
        lines.append("")
        lines.append("Other options:")
        for rid in [r for r in R if r != w]:
            lines.append(f"  {rid} {R[rid]['name']}: violates - {'; '.join(R[rid]['violations'])}.")
        return "\n".join(lines)
    if ru:
        adv, drw = compare(decision, w, ru)
        s = f"Why {w} over {ru} ({R[ru]['name']}): it "
        parts = []
        if drw:
            parts.append(drw[0][2] + (", " + ", ".join(d[2] for d in drw[1:]) if len(drw) > 1 else ""))
        if adv:
            joined = " and ".join(a[2] for a in adv[:2]) if len(adv) <= 2 else \
                ", ".join(a[2] for a in adv[:-1]) + " and " + adv[-1][2]
            parts.append(("but " if drw else "") + joined)
        if not parts:
            parts.append("is marginally better across the board")
        s += ", ".join(parts) + "."
        if drw and adv:
            s += (f" For a {PROFILE_LABEL[decision['profile']]} shipment, the gain in "
                  f"{CRITERION_LABEL[adv[0][1]]} outweighs the {CRITERION_LABEL[drw[0][1]]} penalty")
        else:
            s += f" Under the {PROFILE_LABEL[decision['profile']]} profile this is the stronger option"
        s += (f" (trade-off penalty {rw['score']:.0f} vs {R[ru]['score']:.0f} points; "
              f"0 = best on every criterion).")
        lines.append(s)
    else:
        lines.append("Why: it is the only route that is both feasible and not dominated by another option.")

    others = [r for r in R if r not in (w, ru)]
    if others:
        lines.append("")
        lines.append("Other options:")
        for rid in others:
            r = R[rid]
            if r["violations"] and not decision["fallback"]:
                lines.append(f"  {rid} {r['name']}: ruled out - {'; '.join(r['violations'])}.")
            elif r["dominated_by"]:
                d = r["dominated_by"]
                lines.append(f"  {rid} {r['name']}: never a contender - {d} is at least as good on cost, time, "
                             f"delay risk and geo risk, and strictly better on at least one.")
            elif r["score"] is not None:
                k = max(r["contrib"], key=r["contrib"].get)
                _, drw = compare(decision, rid, w)
                weak = next((d[2] for d in drw if d[1] == k), None) or f"weaker {CRITERION_LABEL[k]}"
                lines.append(f"  {rid} {r['name']}: penalty {r['score']:.0f} - main weakness vs {w}: it {weak}.")
    label, detail = robustness_verdict(sens)
    if label:
        lines.append("")
        lines.append(f"Confidence: {label} - {detail}.")
        flips = [(v["shift"], k, v) for k, v in sens["flips"].items() if v]
        if flips:
            _, k, v = min(flips)
            lines.append(f"  Break-even: switches to {v['new_winner']} ({R[v['new_winner']]['name']}) if the "
                         f"{CRITERION_LABEL[k]} weight rises from {v['old_weight']:.0%} to {v['new_weight']:.0%}.")
        if label == "CLOSE CALL":
            lines.append("  Recommend operator review: the top options are near-equivalent under plausible priorities.")
    return "\n".join(lines)

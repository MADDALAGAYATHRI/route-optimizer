"""Recommend a route for one shipment and explain the decision.

Examples:
  python recommend.py --list
  python recommend.py SHP-012
  python recommend.py SHP-012 --profile cost_first
  python recommend.py --file my_shipment.json      # custom shipment (same schema)
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from routeopt import PROFILES, compute_metrics, decide, explain, load_dataset  # noqa: E402
from routeopt.sensitivity import estimate_stability, flip_points, weight_stability  # noqa: E402


def route_table(decision):
    hdr = f"{'':2}{'ID':4}{'Route':42}{'Cost $':>10}{'E[days]':>9}{'P90':>7}{'Delay%':>8}{'Geo':>6}  Status"
    out = [hdr, "-" * len(hdr) + "-" * 18]
    for rid, r in decision["routes"].items():
        if rid == decision["winner"]:
            status = "RECOMMENDED"
        elif r["violations"] and not decision["fallback"]:
            status = "infeasible"
        elif r["dominated_by"]:
            status = f"dominated by {r['dominated_by']}"
        elif r["score"] is not None:
            status = f"penalty {r['score']:.0f}"
        else:
            status = "violates constraints"
        mark = "*" if rid == decision["winner"] else " "
        out.append(f"{mark:2}{rid:4}{r['name'][:41]:42}{r['cost_usd']:>10,.0f}{r['expected_days']:>9.1f}"
                   f"{r['p90_days']:>7.1f}{r['delay_prob']:>8.0%}{r['geopolitical_risk']:>6.2f}  {status}")
    return "\n".join(out)


def run(shipment, profile=None, sensitivity=True):
    metrics = compute_metrics(shipment)
    d = decide(shipment, profile=profile, metrics=metrics)
    sens = None
    if sensitivity and not d["fallback"]:
        sens = dict(weights=weight_stability(shipment, metrics, d["weights"], d["winner"]),
                    estimates=estimate_stability(shipment, d["weights"], d["winner"]),
                    flips=flip_points(shipment, metrics, d["weights"], d["winner"]))
    return d, sens


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("shipment_id", nargs="?")
    ap.add_argument("--data", default=str(ROOT / "data" / "shipments.json"))
    ap.add_argument("--file", help="JSON file with a single shipment")
    ap.add_argument("--profile", choices=list(PROFILES), help="override the shipment's priority profile")
    ap.add_argument("--no-sensitivity", action="store_true")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()

    if a.file:
        shipment = json.loads(Path(a.file).read_text(encoding="utf-8"))
    else:
        data = load_dataset(a.data)
        if a.list or not a.shipment_id:
            for s in data:
                print(f"{s['shipment_id']}  {s['lane']:22} {s['cargo_type']:12} {s['priority_profile']:14} "
                      f"{len(s['routes'])} routes")
            sys.exit(0)
        shipment = next((s for s in data if s["shipment_id"] == a.shipment_id), None)
        if shipment is None:
            sys.exit(f"Unknown shipment {a.shipment_id}. Use --list.")

    d, sens = run(shipment, a.profile, not a.no_sensitivity)
    print(route_table(d))
    print()
    print(explain(d, shipment, sens))

"""How fragile is a recommendation?

* weight_stability    -- re-decide under random weights drawn from a Dirichlet
                         centred on the profile (concentration 40 -> roughly
                         +/-20% relative jitter per weight). Stability = share
                         of draws that keep the original winner.
* estimate_stability  -- re-simulate with a systematic per-route error on all
                         delay probabilities (log-normal, wider for routes with
                         low-confidence data). Answers: "if our risk estimates
                         are off, does the answer change?"
* profile_sweep       -- winner under each priority profile (shows context
                         sensitivity: the same shipment can have different
                         correct answers).
* flip_points         -- move the weights toward each single criterion and
                         report how far they must move before the winner
                         changes (a "break-even" statement for operators).
"""
from __future__ import annotations

from collections import Counter

import numpy as np

from .scoring import CRITERIA, PROFILES, compute_metrics, decide
from .simulate import confidence_sigma, route_seed


def weight_stability(shipment, metrics, weights, winner, n=400, concentration=40.0, seed=None):
    rng = np.random.default_rng(route_seed(shipment["shipment_id"], "weights") if seed is None else seed)
    alpha = np.array([weights[k] for k in CRITERIA]) * concentration
    counts = Counter()
    for w in rng.dirichlet(alpha, size=n):
        d = decide(shipment, weights=dict(zip(CRITERIA, w)), metrics=metrics)
        counts[d["winner"]] += 1
    return dict(stability=counts[winner] / n, winners={k: v / n for k, v in counts.items()})


def estimate_stability(shipment, weights, winner, reps=40, n_draws=1500, seed=None):
    rng = np.random.default_rng(route_seed(shipment["shipment_id"], "estimates") if seed is None else seed)
    counts = Counter()
    for rep in range(reps):
        bias = {r["route_id"]: float(rng.lognormal(0.0, confidence_sigma(r["estimate_confidence"])))
                for r in shipment["routes"]}
        m = compute_metrics(shipment, salt=rep + 1, bias=bias, n=n_draws)
        counts[decide(shipment, weights=weights, metrics=m)["winner"]] += 1
    return dict(stability=counts[winner] / reps, winners={k: v / reps for k, v in counts.items()})


def profile_sweep(shipment, metrics):
    return {p: decide(shipment, profile=p, metrics=metrics)["winner"] for p in PROFILES}


def flip_points(shipment, metrics, weights, winner, steps=50):
    """For each criterion k: smallest shift t toward 'only k matters' that changes the winner."""
    base = np.array([weights[k] for k in CRITERIA])
    out = {}
    for i, k in enumerate(CRITERIA):
        target = np.zeros(len(CRITERIA))
        target[i] = 1.0
        out[k] = None
        for t in np.linspace(0, 1, steps + 1)[1:]:
            w = (1 - t) * base + t * target
            d = decide(shipment, weights=dict(zip(CRITERIA, w)), metrics=metrics)
            if d["winner"] != winner:
                out[k] = dict(shift=float(t), new_weight=float(w[i]), old_weight=float(base[i]),
                              new_winner=d["winner"])
                break
    return out

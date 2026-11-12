"""Monte Carlo transit-time model.

Each route carries a list of delay *events* (hubs, chokepoints, long legs).
For every simulated voyage:

  1. A route-level condition factor m ~ LogNormal(0, sigma) is drawn. sigma is
     larger when the route's delay statistics are less established
     (low ``confidence``), so uncertain estimates produce fatter tails.
     Because m is shared by all events in the voyage, delays are correlated
     (a bad season hits every hub at once) -- independent per-event noise
     would average out and understate tail risk.
  2. Each event fires with probability min(p * m, 0.95) and adds a
     Gamma(shape=2) delay with mean ``delay_mean * sqrt(m)``.
  3. Geopolitical disruption: each event with geo risk g triggers a rare
     major disruption (reroute / closure) with probability 0.08 * g^2,
     adding 7-15 days.

Outputs per route: expected days, P50/P90 days, P(delay > 1 day) and,
if a deadline is given, P(arrival <= deadline).
"""
from __future__ import annotations

import zlib

import numpy as np

N_DRAWS = 4000


def confidence_sigma(confidence: float) -> float:
    """Map data confidence (pseudo sample size) to log-sd of the condition factor."""
    return float(np.clip(1.6 / np.sqrt(max(confidence, 1.0)), 0.12, 0.50))


def route_seed(shipment_id: str, route_id: str, salt: int = 0) -> int:
    return zlib.crc32(f"{shipment_id}|{route_id}|{salt}".encode())


def simulate_route(route: dict, n: int = N_DRAWS, seed: int = 0,
                   bias: float = 1.0) -> np.ndarray:
    """Return an array of n simulated total transit times (days).

    bias: systematic multiplier on all event probabilities (used for
    estimate-error robustness tests).
    """
    rng = np.random.default_rng(seed)
    sigma = confidence_sigma(route["estimate_confidence"])
    m = rng.lognormal(mean=-0.5 * sigma**2, sigma=sigma, size=n)  # E[m] = 1
    total = np.full(n, route["base_transit_days"], dtype=float)
    for ev in route["events"]:
        p = np.minimum(ev["p"] * bias * m, 0.95)
        fired = rng.random(n) < p
        mean_delay = ev["delay_mean"] * np.sqrt(m)
        delay = rng.gamma(shape=2.0, scale=mean_delay / 2.0)
        total += fired * delay
        if ev["geo"] > 0:
            p_dis = 0.08 * ev["geo"] ** 2 * bias
            dis = rng.random(n) < p_dis
            total += dis * rng.uniform(7, 15, size=n)
    return total


def summarize(samples: np.ndarray, base_days: float, deadline: float | None) -> dict:
    out = dict(
        expected_days=float(samples.mean()),
        p50_days=float(np.percentile(samples, 50)),
        p90_days=float(np.percentile(samples, 90)),
        delay_prob=float(np.mean(samples - base_days > 1.0)),
    )
    out["on_time_prob"] = float(np.mean(samples <= deadline)) if deadline else None
    return out


def route_metrics(shipment: dict, route: dict, n: int = N_DRAWS, salt: int = 0,
                  bias: float = 1.0) -> dict:
    samples = simulate_route(route, n=n, seed=route_seed(shipment["shipment_id"], route["route_id"], salt),
                             bias=bias)
    return summarize(samples, route["base_transit_days"], shipment["constraints"]["deadline_days"])

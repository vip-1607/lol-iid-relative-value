"""Joint long-only Kelly sizing for correlated series markets."""

from dataclasses import dataclass
from math import fsum

import numpy as np
from scipy.optimize import minimize

from .model import SeriesState, probability, terminal_scores
from .portfolio import Quote


@dataclass(frozen=True)
class KellyAllocation:
    full_kelly_fractions: dict[str, float]
    fractional_fractions: dict[str, float]
    stakes: dict[str, float]
    shares: dict[str, float]
    expected_log_growth: float
    outcome_wealth: dict[str, float]
    success: bool
    message: str


def optimize_portfolio_kelly(
    quotes: dict[str, Quote], distribution: dict[tuple[int, int], float],
    state: SeriesState, bankroll: float, fractional_kelly: float = 0.25,
) -> KellyAllocation:
    """Jointly optimize cash fractions; no borrowing, short sales or independent-leg Kelly."""
    if not np.isfinite(bankroll) or bankroll <= 0:
        raise ValueError("bankroll must be positive")
    fractional_kelly = probability(fractional_kelly, "fractional_kelly")
    scores = terminal_scores(state)
    if set(distribution) != set(scores) or abs(fsum(distribution.values()) - 1) > 1e-9:
        raise ValueError("distribution must cover terminal scores and sum to one")
    names = tuple(quotes)
    if not names:
        return KellyAllocation({}, {}, {}, {}, 0.0,
                               {f"{a}:{b}": 1.0 for a, b in scores}, True, "no candidates")
    costs = np.array([quotes[n].unit_cost for n in names], dtype=float)
    payout = np.array([[quotes[n].payout(s) / costs[i] for i, n in enumerate(names)]
                       for s in scores], dtype=float)
    probs = np.array([distribution[s] for s in scores], dtype=float)
    upper = np.array([min(1.0, quotes[n].depth * costs[i] / bankroll)
                      for i, n in enumerate(names)], dtype=float)

    def wealth(x):
        return 1.0 - np.sum(x) + payout @ x

    def objective(x):
        w = wealth(x)
        if np.any(w <= 0):
            return 1e12
        return -float(probs @ np.log(w))

    constraints = ({"type": "ineq", "fun": lambda x: 1.0 - np.sum(x)},)
    result = minimize(objective, np.minimum(upper, 0.01), method="SLSQP",
                      bounds=[(0.0, float(u)) for u in upper], constraints=constraints,
                      options={"ftol": 1e-12, "maxiter": 2000})
    full = np.maximum(0.0, result.x if result.success else np.zeros(len(names)))
    scaled = fractional_kelly * full
    w = wealth(scaled)
    full_map = {n: float(x) for n, x in zip(names, full)}
    frac_map = {n: float(x) for n, x in zip(names, scaled)}
    stakes = {n: bankroll * frac_map[n] for n in names}
    shares = {n: stakes[n] / quotes[n].unit_cost for n in names}
    growth = float(probs @ np.log(w)) if np.all(w > 0) else float("-inf")
    return KellyAllocation(full_map, frac_map, stakes, shares, growth,
                           {f"{s[0]}:{s[1]}": float(v) for s, v in zip(scores, w)},
                           bool(result.success), str(result.message))

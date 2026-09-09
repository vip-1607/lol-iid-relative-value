"""Invert low-degree series polynomials on the complete closed interval [0,1]."""

from dataclasses import dataclass

from numpy.polynomial import Polynomial
from scipy.optimize import brentq

from .model import Market, SeriesState, market_polynomial, probability

VALUE_TOL = 1e-12
ROOT_TOL = 1e-10


def _deduplicate(values):
    result = []
    for value in sorted(float(v) for v in values):
        if not result or abs(value - result[-1]) > ROOT_TOL:
            result.append(value)
    return result


def _roots(poly: Polynomial):
    """Recursive derivative partition catches tangencies and multiple roots.

    Supported model coefficients have degree <=5. Numerical equality uses an
    absolute probability tolerance, not a claim of exact symbolic root finding.
    """
    poly = poly.trim(tol=1e-14)
    if poly.degree() == 0:
        return []
    critical = _roots(poly.deriv())
    bounds = _deduplicate([0.0, *critical, 1.0])
    roots = [x for x in bounds if abs(float(poly(x))) <= VALUE_TOL]
    for lo, hi in zip(bounds, bounds[1:]):
        flo, fhi = float(poly(lo)), float(poly(hi))
        if abs(flo) > VALUE_TOL and abs(fhi) > VALUE_TOL and flo * fhi < 0:
            roots.append(brentq(poly, lo, hi, xtol=1e-14, rtol=1e-14))
    return _deduplicate(roots)


@dataclass(frozen=True)
class InversionResult:
    status: str  # unique, multiple, no_solution, unidentifiable
    roots: tuple[float, ...]
    attainable_min: float
    attainable_max: float
    direction: str  # increasing, decreasing, nonmonotone, constant
    intervals: tuple[tuple[float, float], ...] = ()


def invert_market(market_prob: float, market: Market, state: SeriesState) -> InversionResult:
    target = probability(market_prob, "market_prob")
    poly = market_polynomial(market, state)
    critical = _roots(poly.deriv()) if poly.degree() else []
    bounds = _deduplicate([0.0, *critical, 1.0])
    values = [float(poly(x)) for x in bounds]
    low, high = max(0.0, min(values)), min(1.0, max(values))
    if poly.degree() == 0:
        matched = abs(float(poly(0)) - target) <= VALUE_TOL
        return InversionResult("unidentifiable" if matched else "no_solution", (),
                               low, high, "constant", ((0.0, 1.0),) if matched else ())
    slopes = [float(poly.deriv()((a + b) / 2)) for a, b in zip(bounds, bounds[1:])]
    direction = ("increasing" if all(s >= -VALUE_TOL for s in slopes) else
                 "decreasing" if all(s <= VALUE_TOL for s in slopes) else "nonmonotone")
    roots = tuple(_roots(poly - target))
    if any(abs(float(poly(x)) - target) > 1e-9 for x in roots):
        raise ArithmeticError("Inverse root failed forward validation")
    status = "no_solution" if not roots else "unique" if len(roots) == 1 else "multiple"
    return InversionResult(status, roots, low, high, direction)


def inverse_price_band(bid: float, ask: float, market: Market, state: SeriesState):
    """Return all p intervals compatible with a bid/ask band, not a selected root."""
    bid, ask = probability(bid, "bid"), probability(ask, "ask")
    if bid > ask:
        raise ValueError("bid must not exceed ask")
    poly = market_polynomial(market, state)
    cuts = _deduplicate([0, 1, *invert_market(bid, market, state).roots,
                         *invert_market(ask, market, state).roots])
    inside = lambda x: bid - VALUE_TOL <= float(poly(x)) <= ask + VALUE_TOL
    pieces = [(x, x) for x in cuts if inside(x)]
    pieces += [(a, b) for a, b in zip(cuts, cuts[1:]) if inside((a + b) / 2)]
    merged = []
    for a, b in sorted(pieces):
        if merged and a <= merged[-1][1] + ROOT_TOL:
            merged[-1] = (merged[-1][0], max(b, merged[-1][1]))
        else:
            merged.append((a, b))
    return tuple(merged)


def local_price_sensitivity(p: float, market: Market, state: SeriesState) -> float:
    """d(price)/dp; totals may have the opposite direction to a match market."""
    return float(market_polynomial(market, state).deriv()(probability(p, "p")))

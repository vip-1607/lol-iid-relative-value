"""One absorbing-state DP for match, total-map and handicap probabilities.

All scores and totals refer to the entire series, including completed maps.
Markets assume normal completion and identical settlement rules. Half-point
lines avoid ambiguous push/void rules; exceptional settlements are out of scope.
"""

from dataclasses import dataclass
from math import fsum, isfinite
from numbers import Integral, Real

from numpy.polynomial import Polynomial


def finite_number(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def probability(value, name="probability") -> float:
    value = finite_number(value, name)
    if not 0 <= value <= 1:
        raise ValueError(f"{name} must be in [0, 1]")
    return value


def nonnegative(value, name: str) -> float:
    value = finite_number(value, name)
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def integer(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    return int(value)


@dataclass(frozen=True)
class SeriesState:
    best_of: int = 3
    a: int = 0
    b: int = 0

    def __post_init__(self):
        integer(self.best_of, "best_of")
        integer(self.a, "a")
        integer(self.b, "b")
        if self.best_of not in (3, 5):
            raise ValueError("This research release supports Bo3 and Bo5 only")
        m = self.target
        if not (0 <= self.a <= m and 0 <= self.b <= m) or (self.a == m and self.b == m):
            raise ValueError("Invalid series score")

    @property
    def target(self):
        return self.best_of // 2 + 1

    @property
    def ended(self):
        return self.a == self.target or self.b == self.target

    @classmethod
    def from_target(cls, target, a=0, b=0):
        return cls(2 * integer(target, "target") - 1, a, b)


@dataclass(frozen=True)
class Market:
    """A-oriented YES claim; NO is its exact complement under these rules."""

    kind: str
    line: float | None = None

    def __post_init__(self):
        if self.kind not in {"match", "total_exact", "total_over", "handicap"}:
            raise ValueError("Unknown market kind")
        if self.kind == "match":
            if self.line is not None:
                raise ValueError("match does not take a line")
        elif self.kind == "total_exact":
            if integer(self.line, "total maps") < 0:
                raise ValueError("total maps must be nonnegative")
        else:
            line = finite_number(self.line, "line")
            if not (2 * line).is_integer() or line.is_integer():
                raise ValueError("Use half-point lines; integer-line push rules are unsupported")

    @property
    def label(self):
        return self.kind if self.line is None else f"{self.kind}:{self.line}"

    def payout(self, score: tuple[int, int]) -> int:
        a, b = score
        if self.kind == "match":
            return int(a > b)
        if self.kind == "total_exact":
            return int(a + b == self.line)
        if self.kind == "total_over":
            return int(a + b > self.line)
        return int(a + self.line > b)


def terminal_scores(state: SeriesState) -> tuple[tuple[int, int], ...]:
    if state.ended:
        return ((state.a, state.b),)
    m = state.target
    return tuple([(m, b) for b in range(state.b, m)] +
                 [(a, m) for a in range(state.a, m)])


def _propagate(p, state):
    # The same forward recurrence works for numbers and probability polynomials.
    reach = {(state.a, state.b): p * 0 + 1}
    for a in range(state.a, state.target):
        for b in range(state.b, state.target):
            mass = reach.get((a, b), p * 0)
            for score, weight in [((a + 1, b), p), ((a, b + 1), 1 - p)]:
                reach[score] = reach.get(score, p * 0) + mass * weight
    return {score: reach.get(score, p * 0) for score in terminal_scores(state)}


def terminal_distribution(p: float, state: SeriesState) -> dict[tuple[int, int], float]:
    """Includes structurally reachable outcomes even when their probability is 0."""
    return _propagate(probability(p, "p"), state)


def terminal_polynomials(state: SeriesState) -> dict[tuple[int, int], Polynomial]:
    return _propagate(Polynomial([0.0, 1.0]), state)


def market_polynomial(market: Market, state: SeriesState) -> Polynomial:
    return sum((poly * market.payout(score)
                for score, poly in terminal_polynomials(state).items()), Polynomial([0.0])).trim()


def aggregate(distribution, market: Market) -> float:
    value = fsum(mass * market.payout(score) for score, mass in distribution.items())
    # Clamp only arithmetic round-off of computed probabilities; external input
    # validation remains strict so a malformed quote is never silently repaired.
    if not -1e-12 <= value <= 1 + 1e-12:
        raise ValueError("Aggregated probability lies outside [0,1]")
    return min(1.0, max(0.0, value))


def market_probability(p: float, market: Market, state: SeriesState) -> float:
    return aggregate(terminal_distribution(p, state), market)


def total_distribution(distribution) -> dict[int, float]:
    totals = {}
    for (a, b), mass in distribution.items():
        totals[a + b] = totals.get(a + b, 0.0) + mass
    return dict(sorted(totals.items()))


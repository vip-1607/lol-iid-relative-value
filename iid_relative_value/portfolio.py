"""Long-only, fully funded synthetic portfolios under normal series settlement."""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog

from .model import Market, SeriesState, nonnegative, probability, terminal_scores


@dataclass(frozen=True)
class FeeModel:
    """Research cost per share = rate*price*(1-price) + flat_per_share.

    No exchange rounding, rebates or future API fee schedule is assumed.
    Fees are treated as cash expenses. For historical replay supply the actual
    market/role parameters; zero defaults mean an explicitly frictionless case.
    """
    rate: float = 0.0
    flat_per_share: float = 0.0

    def __post_init__(self):
        nonnegative(self.rate, "fee rate")
        nonnegative(self.flat_per_share, "flat_per_share")

    def per_share(self, price):
        price = probability(price, "price")
        return self.rate * price * (1 - price) + self.flat_per_share


@dataclass(frozen=True)
class Quote:
    market: Market
    side: str
    bid: float
    ask: float
    depth: float
    fees: FeeModel = FeeModel()
    slippage: float = 0.0

    def __post_init__(self):
        if self.side not in {"YES", "NO"}:
            raise ValueError("side must be YES or NO")
        probability(self.bid, "bid")
        probability(self.ask, "ask")
        nonnegative(self.depth, "depth")
        nonnegative(self.slippage, "slippage")
        if self.bid > self.ask or self.ask <= 0 or self.ask + self.slippage > 1:
            raise ValueError("Require 0 < ask <= 1, bid <= ask, ask + slippage <= 1")

    @property
    def execution_price(self):
        return self.ask + self.slippage

    @property
    def unit_cost(self):
        return self.execution_price + self.fees.per_share(self.execution_price)

    def payout(self, score):
        yes = self.market.payout(score)
        return yes if self.side == "YES" else 1 - yes


@dataclass(frozen=True)
class Leg:
    quote: Quote
    shares: float

    def __post_init__(self):
        nonnegative(self.shares, "shares")
        if self.shares > self.quote.depth + 1e-9:
            raise ValueError("Shares exceed quoted depth")


@dataclass(frozen=True)
class PortfolioResult:
    cost: float
    pnl: dict[str, float]
    worst_pnl: float
    best_pnl: float
    guaranteed_positive: bool
    expected_pnl: float | None


def evaluate_portfolio(legs, state: SeriesState, distribution=None) -> PortfolioResult:
    legs = tuple(legs)
    scores = terminal_scores(state)
    cost = sum(leg.shares * leg.quote.unit_cost for leg in legs)
    pnl = {f"{a}:{b}": sum(leg.shares * leg.quote.payout((a, b)) for leg in legs) - cost
           for a, b in scores}
    expected = None
    if distribution is not None:
        if set(distribution) != set(scores):
            raise ValueError("Distribution must cover all structurally reachable terminal scores")
        masses = [probability(distribution[s], "outcome probability") for s in scores]
        if abs(sum(masses) - 1) > 1e-9:
            raise ValueError("Outcome probabilities must sum to one")
        expected = sum(distribution[s] * pnl[f"{s[0]}:{s[1]}"] for s in scores)
    worst, best = min(pnl.values()), max(pnl.values())
    return PortfolioResult(cost, pnl, worst, best, worst > 1e-9, expected)


def maximin_portfolio(quotes, state: SeriesState, budget: float = 1.0):
    """Maximize worst-case net P&L, allowing cash and bounded YES/NO purchases.

    Positive result is a sufficient structural opportunity under the assumptions,
    not an execution guarantee. Zero does not exclude weak arbitrage that is only
    profitable in some outcomes. No probabilities or short sales enter the LP.
    """
    quotes = tuple(quotes)
    budget = nonnegative(budget, "budget")
    if not quotes:
        return (), evaluate_portfolio((), state)
    costs = np.array([q.unit_cost for q in quotes])
    constraints = [[*(costs - np.array([q.payout(s) for q in quotes])), 1.0]
                   for s in terminal_scores(state)]
    constraints.append([*costs, 0.0])
    result = linprog([*np.zeros(len(quotes)), -1.0], A_ub=constraints,
                     b_ub=[*np.zeros(len(constraints) - 1), budget],
                     bounds=[(0, q.depth) for q in quotes] + [(None, None)], method="highs")
    if not result.success:
        raise ArithmeticError(result.message)
    legs = tuple(Leg(q, min(q.depth, max(0.0, float(n))))
                 for q, n in zip(quotes, result.x[:-1]) if n > 1e-9)
    evaluated = evaluate_portfolio(legs, state)
    if evaluated.cost > budget + 1e-7 or evaluated.worst_pnl < -1e-7:
        raise ArithmeticError("Optimizer result failed independent budget/payoff validation")
    return legs, evaluated


def evaluate_fills(legs, filled_shares, state: SeriesState):
    """P&L for actual filled quantities, with no assumed fill of the other legs."""
    legs, filled_shares = tuple(legs), tuple(filled_shares)
    if len(legs) != len(filled_shares):
        raise ValueError("One filled quantity is required for every leg")
    for leg, n in zip(legs, filled_shares):
        nonnegative(n, "filled shares")
        if n > leg.shares:
            raise ValueError("Filled shares exceed requested shares")
    return evaluate_portfolio([Leg(leg.quote, n) for leg, n in zip(legs, filled_shares)], state)

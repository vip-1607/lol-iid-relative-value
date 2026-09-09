"""High-level i.i.d. relative-value analysis."""

from dataclasses import asdict, dataclass

from .inverse import InversionResult, inverse_price_band, invert_market
from .kelly import KellyAllocation, optimize_portfolio_kelly
from .model import Market, SeriesState, market_polynomial, market_probability, terminal_distribution
from .portfolio import Quote, maximin_portfolio


@dataclass(frozen=True)
class AnchorResult:
    status: str
    p: float | None
    p_intervals: tuple[tuple[float, float], ...]
    midpoint_inversion: InversionResult


@dataclass(frozen=True)
class MarketAnalysis:
    name: str
    side: str
    fair_point: float | None
    fair_min: float | None
    fair_max: float | None
    unit_cost: float
    expected_edge: float | None
    worst_edge: float | None
    status: str


@dataclass(frozen=True)
class RelativeValueResult:
    status: str
    anchor: AnchorResult
    markets: tuple[MarketAnalysis, ...]
    structural_status: str
    structural_worst_pnl: float
    structural_legs: tuple[dict, ...]
    kelly: KellyAllocation | None


def invert_anchor(bid: float, ask: float, market: Market, state: SeriesState) -> AnchorResult:
    if bid > ask:
        raise ValueError("anchor bid must not exceed ask")
    middle = invert_market((bid + ask) / 2, market, state)
    intervals = inverse_price_band(bid, ask, market, state)
    if middle.status != "unique" or len(intervals) != 1:
        return AnchorResult("indeterminate", None, intervals, middle)
    return AnchorResult("ready", middle.roots[0], intervals, middle)


def _market_bounds(market: Market, state: SeriesState, intervals):
    poly = market_polynomial(market, state)
    derivative_roots = [float(x.real) for x in poly.deriv().roots()
                        if abs(float(x.imag)) < 1e-9 and 0 <= float(x.real) <= 1]
    values = []
    for lo, hi in intervals:
        points = [lo, hi, *(x for x in derivative_roots if lo <= x <= hi)]
        values.extend(float(poly(x)) for x in points)
    return min(values), max(values)


def price_markets(p: float, markets: dict[str, Market], state: SeriesState) -> dict[str, float]:
    return {name: market_probability(p, market, state) for name, market in markets.items()}


def find_payoff_arbitrage(quotes: dict[str, Quote], state: SeriesState, budget: float):
    legs, result = maximin_portfolio(tuple(quotes.values()), state, budget)
    by_identity = {id(q): name for name, q in quotes.items()}
    described = tuple({"name": by_identity[id(leg.quote)], "shares": leg.shares,
                       "cost": leg.shares * leg.quote.unit_cost} for leg in legs)
    return described, result


def analyze_relative_value(
    anchor_bid: float, anchor_ask: float, anchor_market: Market,
    quotes: dict[str, Quote], state: SeriesState, bankroll: float = 1.0,
    edge_buffer: float = 0.0, fractional_kelly: float = 0.25,
) -> RelativeValueResult:
    if edge_buffer < 0:
        raise ValueError("edge_buffer must be nonnegative")
    anchor = invert_anchor(anchor_bid, anchor_ask, anchor_market, state)
    arb_legs, arb = find_payoff_arbitrage(quotes, state, bankroll)
    structural = "payoff_arbitrage" if arb.guaranteed_positive else "no_payoff_arbitrage"
    if anchor.status != "ready":
        markets = tuple(MarketAnalysis(name, q.side, None, None, None, q.unit_cost,
                                       None, None, "indeterminate") for name, q in quotes.items())
        return RelativeValueResult("indeterminate", anchor, markets, structural,
                                   arb.worst_pnl, arb_legs, None)

    candidates = {}
    analyses = []
    for name, quote in quotes.items():
        yes_point = market_probability(anchor.p, quote.market, state)
        yes_low, yes_high = _market_bounds(quote.market, state, anchor.p_intervals)
        if quote.side == "YES":
            point, low, high = yes_point, yes_low, yes_high
        else:
            point, low, high = 1 - yes_point, 1 - yes_high, 1 - yes_low
        expected_edge, worst_edge = point - quote.unit_cost, low - quote.unit_cost
        status = "actionable_model_edge" if worst_edge > edge_buffer + 1e-12 else (
            "uncertain" if expected_edge > edge_buffer else "no_edge")
        if status == "actionable_model_edge":
            candidates[name] = quote
        analyses.append(MarketAnalysis(name, quote.side, point, low, high, quote.unit_cost,
                                       expected_edge, worst_edge, status))
    distribution = terminal_distribution(anchor.p, state)
    kelly = optimize_portfolio_kelly(candidates, distribution, state, bankroll, fractional_kelly)
    overall = "model_edge" if candidates else "no_model_edge"
    return RelativeValueResult(overall, anchor, tuple(analyses), structural,
                               arb.worst_pnl, arb_legs, kelly)


def result_as_dict(result: RelativeValueResult):
    return asdict(result)

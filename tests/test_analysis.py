import pytest

from iid_relative_value import FeeModel, Market, Quote, SeriesState, analyze_relative_value


def quote(market, side, ask, depth=100, fee=0, slippage=0):
    return Quote(market, side, max(0, ask - .01), ask, depth, FeeModel(fee), slippage)


def test_match_anchor_and_yes_no_edges_with_costs():
    state = SeriesState(3)
    quotes = {
        "sweep yes": quote(Market("handicap", -1.5), "YES", .30),
        "sweep no": quote(Market("handicap", -1.5), "NO", .70),
        "cost removes edge": quote(Market("total_exact", 3), "YES", .45, fee=.5),
    }
    result = analyze_relative_value(.64, .66, Market("match"), quotes, state,
                                     bankroll=1000, edge_buffer=.001)
    assert result.anchor.status == "ready"
    assert result.anchor.p_intervals[0][0] < result.anchor.p < result.anchor.p_intervals[0][1]
    rows = {row.name: row for row in result.markets}
    assert rows["sweep yes"].status == "actionable_model_edge"
    assert rows["sweep no"].status == "no_edge"
    assert rows["cost removes edge"].status != "actionable_model_edge"
    assert result.kelly.success
    assert sum(result.kelly.fractional_fractions.values()) <= .25 + 1e-9
    assert result.kelly.stakes["sweep yes"] > 0


def test_ambiguous_anchor_stops_model_decision():
    result = analyze_relative_value(.47, .49, Market("total_exact", 3),
                                    {"x": quote(Market("match"), "YES", .5)}, SeriesState(3))
    assert result.status == "indeterminate"
    assert len(result.anchor.p_intervals) == 2
    assert result.kelly is None


def test_structural_arbitrage_is_separate_and_depth_bounded():
    market = Market("match")
    quotes = {"yes": quote(market, "YES", .45, depth=1),
              "no": quote(market, "NO", .45, depth=1)}
    result = analyze_relative_value(.49, .51, market, quotes, SeriesState(3), bankroll=1)
    assert result.structural_status == "payoff_arbitrage"
    assert result.structural_worst_pnl == pytest.approx(.1)
    assert sum(leg["shares"] for leg in result.structural_legs) == pytest.approx(2)


def test_fractional_kelly_respects_depth():
    quotes = {"yes": quote(Market("match"), "YES", .4, depth=.1)}
    result = analyze_relative_value(.69, .71, Market("match"), quotes, SeriesState(3), bankroll=100)
    assert result.kelly.shares["yes"] <= .1 + 1e-9

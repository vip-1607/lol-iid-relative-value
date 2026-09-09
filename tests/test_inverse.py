import pytest

from iid_relative_value.inverse import invert_market, inverse_price_band
from iid_relative_value.model import Market, SeriesState, market_probability


STATES = [SeriesState(n, a, b) for n in (3, 5) for a in range(n // 2 + 2)
          for b in range(n // 2 + 2) if not a == b == n // 2 + 1]


@pytest.mark.parametrize("state", STATES)
@pytest.mark.parametrize("p", [0, .2, .5, .7, 1])
def test_roundtrip(state, p):
    for market in [Market("match"), Market("handicap", -1.5),
                   *[Market("total_exact", k) for k in range(state.best_of + 1)]]:
        result = invert_market(market_probability(p, market, state), market, state)
        assert result.status != "no_solution"
        if result.status != "unidentifiable":
            assert any(abs(root - p) < 1e-8 for root in result.roots)


def test_multiple_roots_and_band_are_never_collapsed():
    result = invert_market(.4, Market("total_exact", 5), SeriesState(5, 1, 0))
    assert result.status == "multiple"
    assert len(result.roots) == 2
    assert all(root < .5 for root in result.roots)
    bands = inverse_price_band(.47, .49, Market("total_exact", 3), SeriesState(3))
    assert len(bands) == 2 and bands[0][1] < .5 < bands[1][0]


def test_endpoints_and_absorbed_state():
    assert invert_market(0, Market("match"), SeriesState(3)).roots == pytest.approx((0,))
    assert invert_market(1, Market("match"), SeriesState(3)).roots == pytest.approx((1,))
    assert invert_market(1, Market("match"), SeriesState(3, 2, 0)).status == "unidentifiable"

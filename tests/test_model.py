from itertools import product

import pytest

from iid_relative_value.model import (Market, SeriesState, aggregate, market_probability,
                                      terminal_distribution, terminal_polynomials, terminal_scores)


def states():
    return [SeriesState(n, a, b) for n in (3, 5) for a in range(n // 2 + 2)
            for b in range(n // 2 + 2) if not a == b == n // 2 + 1]


def enumerate_paths(p, state):
    out = {score: 0.0 for score in terminal_scores(state)}
    if state.ended:
        out[state.a, state.b] = 1.0
        return out
    seen = set()
    for sequence in product((1, 0), repeat=state.best_of - state.a - state.b):
        a, b, wins, losses, prefix = state.a, state.b, 0, 0, []
        for result in sequence:
            prefix.append(result); wins += result; losses += 1 - result
            a += result; b += 1 - result
            if a == state.target or b == state.target:
                key = tuple(prefix)
                if key not in seen:
                    out[a, b] += p ** wins * (1 - p) ** losses
                    seen.add(key)
                break
    return out


@pytest.mark.parametrize("state", states())
@pytest.mark.parametrize("p", [0, .01, .2, .5, .7, .99, 1])
def test_dp_against_independent_enumeration_and_symmetry(state, p):
    dist = terminal_distribution(p, state)
    assert dist == pytest.approx(enumerate_paths(p, state), abs=1e-12)
    assert sum(dist.values()) == pytest.approx(1)
    assert {s: float(poly(p)) for s, poly in terminal_polynomials(state).items()} == pytest.approx(dist)
    mirrored = terminal_distribution(1 - p, SeriesState(state.best_of, state.b, state.a))
    assert dist == pytest.approx({(b, a): v for (a, b), v in mirrored.items()})
    for market in [Market("match"), Market("handicap", -1.5),
                   *[Market("total_exact", k) for k in range(state.best_of + 1)]]:
        assert market_probability(p, market, state) == pytest.approx(aggregate(dist, market))


def test_known_prices_and_absorption():
    state = SeriesState(3)
    assert market_probability(.6, Market("match"), state) == pytest.approx(.648)
    assert market_probability(.6, Market("total_exact", 3), state) == pytest.approx(.48)
    assert market_probability(.6, Market("handicap", -1.5), state) == pytest.approx(.36)
    assert terminal_distribution(.5, SeriesState(3, 2, 1)) == {(2, 1): 1.0}


@pytest.mark.parametrize("args", [(4, 0, 0), (3, 2, 2), (3, -1, 0), (5, 4, 0)])
def test_invalid_states(args):
    with pytest.raises(ValueError):
        SeriesState(*args)

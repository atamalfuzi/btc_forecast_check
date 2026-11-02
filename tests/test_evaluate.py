"""The tests and scores, on made-up data where the right answer is known. No downloads."""
import numpy as np
import pandas as pd
import pytest

import data
import evaluate
import forecast
from test_leak import candles


def test_dm_is_zero_for_equal_losses_and_flips_sign():
    rng = np.random.default_rng(0)
    a, b = rng.normal(1, 1, 500), rng.normal(1, 1, 500)
    assert evaluate.dm_test(a, b)[0] == pytest.approx(-evaluate.dm_test(b, a)[0])
    assert evaluate.dm_test(a, b)[1] == pytest.approx(evaluate.dm_test(b, a)[1])


def test_dm_matches_a_hand_calculation_without_autocorrelation():
    """With the lags cancelled out, the statistic is the t-ratio on the mean loss difference."""
    d = np.array([1.0, -1.0] * 200) + 0.1
    statistic, _ = evaluate.dm_test(d, np.zeros(400))
    lrv = evaluate.long_run_variance(d)
    assert statistic == pytest.approx(0.1 / np.sqrt(lrv / 400) * np.sqrt(399 / 400))


def test_dm_size_and_power():
    rng = np.random.default_rng(1)
    rejected = sum(evaluate.dm_test(rng.normal(size=500) ** 2, rng.normal(size=500) ** 2)[1] < 0.05 for _ in range(400))
    assert rejected / 400 < 0.09
    assert evaluate.dm_test(rng.normal(0, 1.3, 500) ** 2, rng.normal(0, 1, 500) ** 2)[1] < 0.01


def test_clark_west_hand_example():
    actual, big = np.array([1.0, -1.0, 2.0, -2.0]), np.array([0.5, -0.5, 1.0, -1.0])
    # the adjustment cancels the noise term: f = 2 * actual * big for a no-change smaller model
    f = 2 * actual * big
    statistic, _ = evaluate.clark_west(actual, np.zeros(4), big)
    assert statistic == pytest.approx(f.mean() / np.sqrt(evaluate.long_run_variance(f) / 4))


@pytest.fixture(scope="module")
def noise():
    return forecast.build(candles(n=900, seed=5), first=300, step=42)


def test_no_model_beats_a_random_walk_on_noise(noise):
    f = noise.dropna(subset=["r_next"])
    for col in ("ols", "arima", "gbm"):
        assert evaluate.dm_test((f.r_next - f[col]) ** 2, f.r_next ** 2)[0] > -1.96


def test_a_real_signal_is_found():
    """If next-day returns could be forecast, this setup would show it."""
    f = forecast.build(candles(n=900, seed=6, ar=0.4), first=300, step=42).dropna(subset=["r_next"])
    for col in ("ols", "arima"):
        statistic, p = evaluate.dm_test((f.r_next - f[col]) ** 2, f.r_next ** 2)
        assert statistic < 0 and p < 0.05
    assert ((f.ols > 0) == (f.r_next > 0)).mean() > 0.56


def test_calibration_uses_only_past_outcomes_and_matches_frequencies():
    rng = np.random.default_rng(7)
    index = pd.date_range("2018-01-01", periods=3000)
    score = pd.Series(rng.normal(size=3000), index=index)
    up = pd.Series((rng.random(3000) < 1 / (1 + np.exp(-score * 0.5))).astype(float), index=index)
    p = evaluate.calibrate(score * 0.01, up, burn=500)
    assert p.iloc[:500].isna().all()
    high = p.iloc[500:] > 0.6
    assert abs(up.iloc[500:][high].mean() - p.iloc[500:][high].mean()) < 0.04
    # changing a later outcome leaves earlier probabilities alone
    later = up.copy()
    later.iloc[2000:] = 1 - later.iloc[2000:]
    pd.testing.assert_series_equal(p.iloc[:2000], evaluate.calibrate(score * 0.01, later, burn=500).iloc[:2000])


def test_climatology_is_the_share_of_earlier_days():
    up = pd.Series([1.0, 0.0, 1.0, 1.0])
    assert evaluate.climatology(up).tolist()[1:] == [1.0, 0.5, 2 / 3]


def test_score_tables_have_the_benchmark_at_zero_skill():
    f = forecast.build(candles(n=1200, seed=8), first=400, step=63)
    f.index = pd.date_range("2016-01-10", periods=len(f), name="date")
    scored = evaluate.prepare(f)
    returns, direction = evaluate.return_scores(scored), evaluate.direction_scores(scored)
    assert returns.oos_r2[0] == 0
    assert returns.price_r2[0] > 0.5  # a no-change forecast looks good on price levels
    assert set(direction.model) == {"Share of up days so far", "ARIMA(4,1,1)", "Lagged regression", "Logit", "Gradient boosting"}


@pytest.mark.parametrize("change", [
    lambda d: d.drop(d.index[50]),
    lambda d: d.assign(volume=d.volume.where(d.index != d.index[50], 0.0)),
    lambda d: d.assign(high=d.high.where(d.index != d.index[50], 1.0)),
])
def test_broken_files_are_refused(change):
    with pytest.raises(ValueError):
        data.check(change(candles()))

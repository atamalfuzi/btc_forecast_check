"""No forecast may use anything from after the moment it is made. Made-up data, no downloads."""
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import r2_score

import forecast
import leak

N = 420


def candles(n=N, seed=0, ar=0.0):
    """A made-up price series with a consistent open, high, low, close and volume each day."""
    rng = np.random.default_rng(seed)
    noise = rng.standard_t(4, n) * 0.03
    r = np.zeros(n)
    for t in range(1, n):
        r[t] = ar * r[t - 1] + noise[t]
    close = 1000 * np.exp(np.cumsum(r))
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) * np.exp(rng.exponential(0.01, n))
    low = np.minimum(open_, close) * np.exp(-rng.exponential(0.01, n))
    index = pd.date_range("2018-01-01", periods=n, name="date")
    return pd.DataFrame(dict(open=open_, high=high, low=low, close=close, volume=rng.lognormal(8, 0.5, n)), index=index)


def corrupt_after(df, t, seed=1):
    """Replace every value after row t with something else, keeping it positive."""
    out = df.copy()
    shock = np.exp(np.random.default_rng(seed).normal(0, 1, (len(df) - t - 1, df.shape[1])))
    out.iloc[t + 1:] = out.iloc[t + 1:].values * shock
    return out


def looks_ahead(build, df, cuts=(100, 250, 400)):
    """Column names whose values up to a cut change when only rows after that cut change."""
    found = set()
    for t in cuts:
        clean, changed = build(df).iloc[:t + 1], build(corrupt_after(df, t)).iloc[:t + 1]
        for col in clean:
            if not np.allclose(clean[col], changed[col], equal_nan=True, rtol=1e-12, atol=1e-12):
                found.add(col)
    return found


def test_no_feature_uses_future_data():
    assert looks_ahead(forecast.features, candles()) == set()


def test_prior_day_inputs_use_no_future_data():
    assert looks_ahead(leak.prior_day_returns, candles()) == set()


@pytest.mark.parametrize("name, build", [
    ("next return", lambda df: pd.DataFrame({"next return": np.log(df.close).diff().shift(-1)})),
    ("centred average", lambda df: pd.DataFrame({"centred average": df.close.rolling(7, center=True).mean()})),
    ("scaled on the whole sample", lambda df: pd.DataFrame({"scaled on the whole sample": df.close / df.close.max()})),
])
def test_the_check_catches_features_that_do_leak(name, build):
    """The check above has to be able to fail. Corrupting the far future changes all of these."""
    assert name in looks_ahead(build, candles())


def test_row_t_inputs_do_not_include_the_day_they_forecast():
    """Corrupt day t alone: row t of the prior-day inputs, which forecast day t, must not move."""
    df = candles()
    t = 300
    changed = df.copy()
    changed.iloc[t] = changed.iloc[t].values * 1.5
    assert np.allclose(leak.prior_day(df).iloc[t], leak.prior_day(changed).iloc[t])
    assert np.allclose(leak.prior_day_returns(df).iloc[t], leak.prior_day_returns(changed).iloc[t], equal_nan=True)
    assert not np.allclose(leak.same_day(df).iloc[t], leak.same_day(changed).iloc[t])


def test_target_is_the_next_days_return():
    df = candles()
    assert np.isclose(forecast.target(df).iloc[10], np.log(df.close.iloc[11] / df.close.iloc[10]))
    assert np.isnan(forecast.target(df).iloc[-1])


def test_training_rows_are_all_in_the_past():
    """Rows before the block start have known targets. The spy records what each fit was shown."""
    seen = []

    class Spy:
        def fit(self, X, y):
            seen.append(X[:, 0].max())
            return self

        def predict(self, X):
            seen.append(("predict", X[:, 0].min()))
            return np.zeros(len(X))

    X = pd.DataFrame({"row": np.arange(100.0)})
    forecast.walk_forward(X, pd.Series(np.arange(100.0)), Spy, first=50, step=10)
    for trained, (_, predicted) in zip(seen[::2], seen[1::2]):
        assert trained < predicted


@pytest.mark.parametrize("window", [None, 200])
def test_every_forecast_is_unchanged_when_later_prices_change(window):
    """End to end, including ARIMA's filter and each model's scaler: only the past can move a forecast."""
    df = candles()
    t = 330
    a = forecast.build(df, first=250, window=window)
    b = forecast.build(corrupt_after(df, t), first=250, window=window)
    cols = ["arima", "ols", "gbm", "logit_p", "gbm_p"]
    pd.testing.assert_frame_equal(a.loc[:df.index[t], cols], b.loc[:df.index[t], cols], rtol=1e-6)
    # and the corruption did reach the models, so the comparison means something
    assert not np.allclose(a[cols].iloc[-1], b[cols].iloc[-1])


def test_same_day_features_score_high_and_prior_day_features_do_not():
    """On prices that can't be forecast, same-day inputs still look excellent."""
    df = candles(n=1000, seed=3)
    r = np.log(df.close).diff().iloc[40:].values
    train, test = leak.holdout(len(r))
    same = leak.same_day_returns(df).iloc[40:].values
    prior = leak.prior_day_returns(df).iloc[40:].values
    assert r2_score(*leak.fit_score(same, r, train, test, leak.StandardScaler)) > 0.8
    assert r2_score(*leak.fit_score(prior, r, train, test, leak.StandardScaler)) < 0.05


def test_leak_table_reproduces_the_pattern():
    df = candles(n=1900, seed=4)
    df.index = pd.date_range("2016-10-01", periods=1900, name="date")
    table = leak.leak_table(df).set_index("step")
    assert table.r2[1] > 0.99  # random holdout, whole-sample scaling
    assert table.r2[6] > 0.8 and table.r2[7] < 0.05  # the same split in returns

"""Walk-forward forecasts of the next day's log return. Row t holds what is known at the close of day t."""
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA

from data import DATA, load

FIRST_FORECAST = "2016-01-10"  # one year of training after the sample starts
STEP = 21  # days between refits; forecasts in between use today's features with the last fitted parameters
ROLLING = 730  # days of training in the rolling-window check
ORDER = (4, 1, 1)  # the paper's first ARIMA


def features(df):
    """Lagged returns, the day's range and volume, 30-day volatility and momentum. Nothing here uses later rows."""
    r = np.log(df.close).diff()
    X = pd.DataFrame({f"ret_{k}": r.shift(k) for k in range(5)})
    X["range"] = np.log(df.high / df.low)
    log_volume = np.log(df.volume)
    X["rel_volume"] = log_volume - log_volume.rolling(30).mean()
    X["vol_30"] = r.rolling(30).std()
    X["mom_30"] = r.rolling(30).sum()
    return X


def target(df):
    """Row t's target is the return from the close of t to the close of t+1, so it is only known a day later."""
    return np.log(df.close).diff().shift(-1)


def walk_forward(X, y, model, first, step=STEP, window=None, proba=False):
    """Refit every `step` rows. Rows before `start` have targets known by the close of `start`, so they can train it.

    The scaler sits inside each model's pipeline, so it is refit on the training rows alone.
    """
    out = pd.Series(np.nan, index=X.index)
    for start in range(first, len(X), step):
        lo = 0 if window is None else max(0, start - window)
        fitted = model().fit(X.iloc[lo:start].values, y.iloc[lo:start].values)
        block = X.iloc[start:start + step].values
        out.iloc[start:start + step] = fitted.predict_proba(block)[:, 1] if proba else fitted.predict(block)
    return out


def arima_forecasts(close, first, step=STEP, window=None, order=ORDER):
    """ARIMA on log price, as in the paper. A fixed set of parameters is run forward between refits."""
    log_close, out, params = np.log(close.values), np.full(len(close), np.nan), None
    for start in range(first, len(close), step):
        lo = 0 if window is None else max(0, start - window)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fitted = ARIMA(log_close[lo:start + 1], order=order).fit(start_params=params)
        params = fitted.params
        end = min(start + step, len(close))
        # One-step forecasts of days start+1 to end. Each uses only the days before the one it forecasts.
        predicted = fitted.apply(log_close[lo:end]).predict(start=start + 1 - lo, end=end - lo)
        out[start:end] = predicted - log_close[start:end]
    return pd.Series(out, index=close.index)


def ols():
    return make_pipeline(StandardScaler(), LinearRegression())


def logit():
    return make_pipeline(StandardScaler(), LogisticRegression())


# Settings are fixed in advance and not tuned, so nothing here was chosen by looking at the scored period.
def gbm():
    return HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=200, min_samples_leaf=50, random_state=0)


def gbm_direction():
    return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, min_samples_leaf=50, random_state=0)


def build(df, first, step=STEP, window=None):
    """All models' forecasts, one row per forecast day. r_next is blank on the last row until the next close."""
    X = features(df).dropna()
    df, y = df.loc[X.index], target(df).loc[X.index]
    up = (y > 0).astype(float).where(y.notna())
    args = dict(first=first, step=step, window=window)
    out = pd.DataFrame({
        "close": df.close,
        "r_next": y,
        "arima": arima_forecasts(df.close, **args),
        "ols": walk_forward(X, y, ols, **args),
        "gbm": walk_forward(X, y, gbm, **args),
        "logit_p": walk_forward(X, up, logit, proba=True, **args),
        "gbm_p": walk_forward(X, up, gbm_direction, proba=True, **args),
    })
    return out.iloc[first:]


if __name__ == "__main__":
    df = load()
    first = len(features(df).dropna().loc[:FIRST_FORECAST]) - 1
    build(df, first).to_csv(DATA / "forecasts.csv")
    build(df, first, window=ROLLING).to_csv(DATA / "forecasts_rolling.csv")

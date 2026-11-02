"""Show the paper's 0.9994 appearing, then going away once only earlier information is used.

python leak.py                     Bitstamp data, written to output/leak.csv and output/scales.csv
python leak.py path/to/btc.csv     a Yahoo-format file, such as the paper's, written to output/*_original.csv
"""
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from data import OUTPUT, load
from forecast import arima_forecasts, features

WINDOW = ("2016-12-21", "2021-12-21")  # the dates the paper gives
HOLDOUTS = 200  # the paper's random seed is unknown, so average over many draws
SHARE = 0.9


def same_day(df):
    """The paper's fitted equation: the close from that same day's open, high, low and volume."""
    return df[["open", "high", "low", "volume"]]


def prior_day(df):
    """Yesterday's values, all of them known before today's close."""
    return df[["open", "high", "low", "close", "volume"]].shift(1)


def same_day_returns(df):
    """The same-day features with price trend removed: where the day's open, high and low sit against yesterday's close."""
    previous = df.close.shift(1)
    return pd.DataFrame({"open": np.log(df.open / previous), "high": np.log(df.high / previous),
                         "low": np.log(df.low / previous), "volume": features(df).rel_volume})


def prior_day_returns(df):
    """The features the walk-forward models use, as they stood the day before."""
    return features(df).shift(1)


def fit_score(X, y, train, test, scaler):
    """Fit the scaler on the training rows and regress. Returns the test targets and predictions."""
    scale = scaler().fit(X[train])
    fitted = LinearRegression().fit(scale.transform(X[train]), y[train])
    return y[test], fitted.predict(scale.transform(X[test]))


def holdout(n):
    cut = int(n * SHARE)
    return np.arange(cut), np.arange(cut, n)


def mean_scores(X, y, splits):
    """Mean R-squared and root of mean squared error across train and test splits."""
    scores = []
    for train, test in splits:
        predicted = LinearRegression().fit(X[train], y[train]).predict(X[test])
        scores.append((r2_score(y[test], predicted), mean_squared_error(y[test], predicted)))
    r2, mse = np.mean(scores, axis=0)
    return r2, mse ** 0.5


def levels_all_scaled(d):
    """The paper's setup: every column min-max scaled on the whole sample before any split."""
    scaled = pd.DataFrame(MinMaxScaler().fit_transform(d), index=d.index, columns=d.columns)
    return scaled[["open", "high", "low", "volume"]].values, scaled.close.values


def leak_table(df):
    d = df.loc[WINDOW[0]:WINDOW[1]]
    rows = []

    def add(step, inputs, target_name, validation, y, p):
        rows.append(dict(step=step, inputs=inputs, target=target_name, validation=validation,
                         r2=r2_score(y, p), rmse=mean_squared_error(y, p) ** 0.5))

    X, y = levels_all_scaled(d)
    draws = [train_test_split(np.arange(len(y)), test_size=1 - SHARE, random_state=seed) for seed in range(HOLDOUTS)]
    for step, validation, splits in ((1, f"random 10%, {HOLDOUTS} draws", draws), (2, "10 unshuffled folds", KFold(10).split(X))):
        r2, rmse = mean_scores(X, y, splits)
        rows.append(dict(step=step, inputs="same day", target="price, min-max scaled", validation=validation, r2=r2, rmse=rmse))

    y = d.close.values
    train, test = holdout(len(y))
    add(3, "same day", "price, dollars", "last 10%, scaled on past", *fit_score(same_day(d).values, y, train, test, MinMaxScaler))
    lagged = prior_day(df).loc[d.index]
    add(4, "day before", "price, dollars", "last 10%, scaled on past", *fit_score(lagged.values, y, train, test, MinMaxScaler))
    add(5, "no-change forecast", "price, dollars", "last 10%", y[test], lagged.close.values[test])

    r = np.log(df.close).diff().loc[d.index].values
    add(6, "same day", "log return", "last 10%, scaled on past",
        *fit_score(same_day_returns(df).loc[d.index].values, r, train, test, StandardScaler))
    add(7, "day before", "log return", "last 10%, scaled on past",
        *fit_score(prior_day_returns(df).loc[d.index].values, r, train, test, StandardScaler))
    add(8, "no-change forecast", "log return", "last 10%", r[test], np.zeros(len(test)))
    return pd.DataFrame(rows)


def scale_table(df):
    """One forecast, the no-change forecast on the paper's test days, measured on each of the paper's scales."""
    d = df.loc[WINDOW[0]:WINDOW[1]]
    test = int(len(d) * SHARE)
    error = d.close.diff().values[test:]
    log_error = np.log(d.close).diff().values[test:]
    arima = arima_forecasts(d.close, first=test - 1, step=1).values[test - 1:-1]
    return pd.DataFrame([
        ("No-change forecast, price in dollars", (error ** 2).mean()),
        ("No-change forecast, price min-max scaled", ((error / (d.close.max() - d.close.min())) ** 2).mean()),
        ("No-change forecast, price min-max scaled, all the paper's days", ((d.close.diff() / (d.close.max() - d.close.min())) ** 2).mean()),
        ("No-change forecast, log return", (log_error ** 2).mean()),
        ("ARIMA(4,1,1) refit daily, log return", ((log_error - arima) ** 2).mean()),
        ("Forecasting one half for an up or down label", 0.25),
    ], columns=["forecast and scale", "mse"])


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else None
    df = load(path) if path else load()
    tag = "_original" if path else ""
    OUTPUT.mkdir(exist_ok=True)
    leak_table(df).round(6).to_csv(OUTPUT / f"leak{tag}.csv", index=False)
    scale_table(df).to_csv(OUTPUT / f"scales{tag}.csv", index=False, float_format="%.6g")

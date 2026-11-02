"""Score the forecasts on one scale against a random walk, and direction against the past share of up days."""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from data import DATA, OUTPUT
from forecast import walk_forward

SCORED_FROM = "2017-01-10"  # a year after the first forecast, so calibration has a year of past forecasts to learn from
BURN = 365
PAPER_TEST = ("2021-06-21", "2021-12-20")  # the last 10% of the paper's dates, by the day each forecast is made
PAPER_FILE_END = "2022-10-31"  # forecasts of the last day in the paper's btc.csv are made here
RETURNS = {"arima": "ARIMA(4,1,1)", "ols": "Lagged regression", "gbm": "Gradient boosting"}
SCORES = {"arima": "ARIMA(4,1,1)", "ols": "Lagged regression", "logit": "Logit", "gbm_p": "Gradient boosting"}


def long_run_variance(d):
    """Newey-West, since squared errors bunch together when volatility does."""
    d, T = d - d.mean(), len(d)
    lags = int(4 * (T / 100) ** (2 / 9))
    variance = d @ d / T
    for k in range(1, lags + 1):
        variance += 2 * (1 - k / (lags + 1)) * (d[k:] @ d[:-k]) / T
    return variance


def dm_test(loss_a, loss_b):
    """Diebold-Mariano for one-step forecasts with the Harvey-Leybourne-Newbold correction. Positive means a is worse."""
    d = np.asarray(loss_a, float) - np.asarray(loss_b, float)
    T = len(d)
    statistic = d.mean() / np.sqrt(long_run_variance(d) / T) * np.sqrt((T - 1) / T)
    return statistic, 2 * stats.t.sf(abs(statistic), T - 1)


def clark_west(actual, small, big):
    """Clark-West test that the bigger model adds something to a smaller one nested in it. One-sided p-value."""
    actual, small, big = (np.asarray(v, float) for v in (actual, small, big))
    f = (actual - small) ** 2 - ((actual - big) ** 2 - (small - big) ** 2)
    statistic = f.mean() / np.sqrt(long_run_variance(f) / len(f))
    return statistic, stats.norm.sf(statistic)


def calibrate(score, up, burn=BURN):
    """Turn a model's forecasts into probabilities with a logit fit on its earlier forecasts and outcomes only."""
    return walk_forward(score.to_frame(), up, lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1e6)),
                        first=burn, proba=True)


def climatology(up):
    """The chance of an up day if all you use is the share of up days so far."""
    return up.expanding().mean().shift(1)


def prepare(f):
    """Add direction scores and calibrated probabilities, then keep the days everything can be scored on."""
    f = f.dropna(subset=["r_next"]).copy()
    f["up"] = (f.r_next > 0).astype(float)
    f["logit"] = np.log(f.logit_p / (1 - f.logit_p))
    f["gbm_score"] = np.log(f.gbm_p.clip(1e-6, 1 - 1e-6) / (1 - f.gbm_p.clip(1e-6, 1 - 1e-6)))
    for col, source in (("arima", "arima"), ("ols", "ols"), ("logit", "logit"), ("gbm_p", "gbm_score")):
        f[f"cal_{col}"] = calibrate(f[source], f.up)
    f["base"] = climatology(f.up)
    return f.loc[SCORED_FROM:]


def return_scores(f):
    rows, T = [], len(f)
    price = f.close * np.exp(f.r_next)
    rw_loss = f.r_next ** 2
    for name, label in {None: "Random walk", **RETURNS}.items():
        forecast = f[name] if name else pd.Series(0.0, index=f.index)
        error, implied = f.r_next - forecast, f.close * np.exp(forecast)
        row = dict(model=label, days=T, rmse=(error ** 2).mean() ** 0.5, mae=error.abs().mean(),
                   oos_r2=1 - (error ** 2).sum() / rw_loss.sum(),
                   price_rmse=((price - implied) ** 2).mean() ** 0.5,
                   price_mape=(abs(price - implied) / price).mean() * 100,
                   price_r2=1 - ((price - implied) ** 2).sum() / ((price - price.mean()) ** 2).sum())
        if name:
            row["dm"], row["dm_p"] = dm_test(error ** 2, rw_loss)
            row["dm_abs"], row["dm_abs_p"] = dm_test(error.abs(), f.r_next.abs())
            row["cw_p"] = clark_west(f.r_next, np.zeros(T), forecast)[1]
        rows.append(row)
    return pd.DataFrame(rows)


def direction_scores(f):
    base_hit = ((f.base > 0.5) == f.up).astype(float)
    base_brier = (f.base - f.up) ** 2
    rows = [dict(model="Share of up days so far", days=len(f), accuracy=base_hit.mean() * 100, calibrated_accuracy=base_hit.mean() * 100,
                 auc=roc_auc_score(f.up, f.base), brier=base_brier.mean())]
    for col, label in SCORES.items():
        raw = f[col] > (0.5 if col == "gbm_p" else 0)
        p = f[f"cal_{col}"]
        hit, brier = (p > 0.5).eq(f.up.astype(bool)).astype(float), (p - f.up) ** 2
        rows.append(dict(model=label, days=len(f), accuracy=raw.eq(f.up.astype(bool)).mean() * 100, calibrated_accuracy=hit.mean() * 100,
                         auc=roc_auc_score(f.up, f[col]), brier=brier.mean(),
                         hit_p=dm_test(1 - hit, 1 - base_hit)[1], brier_dm=dm_test(brier, base_brier)[0],
                         brier_p=dm_test(brier, base_brier)[1]))
    return pd.DataFrame(rows)


def periods(f):
    spans = {"All scored days": (None, None), "Paper's test days": PAPER_TEST,
             "Through the paper's file": (None, PAPER_FILE_END),
             "After the paper's file": (pd.Timestamp(PAPER_FILE_END) + pd.Timedelta(days=1), None)}
    rows = []
    for label, (a, b) in spans.items():
        s = f.loc[a:b]
        for name, model in RETURNS.items():
            statistic, p = dm_test((s.r_next - s[name]) ** 2, s.r_next ** 2)
            rows.append(dict(period=label, days=len(s), model=model, oos_r2=1 - ((s.r_next - s[name]) ** 2).sum() / (s.r_next ** 2).sum(),
                             dm=statistic, dm_p=p))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    f = prepare(pd.read_csv(DATA / "forecasts.csv", parse_dates=["date"], index_col="date"))
    f.to_csv(DATA / "scored.csv")
    rolling = prepare(pd.read_csv(DATA / "forecasts_rolling.csv", parse_dates=["date"], index_col="date"))
    OUTPUT.mkdir(exist_ok=True)
    for name, table in {"returns": return_scores(f), "direction": direction_scores(f), "periods": periods(f),
                        "returns_rolling": return_scores(rolling), "direction_rolling": direction_scores(rolling)}.items():
        table.to_csv(OUTPUT / f"{name}.csv", index=False, float_format="%.6g")
        print(name, table.round(4).to_string(), "", sep="\n")

"""Draw the charts from the saved forecasts."""
import textwrap

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

import leak
from data import DATA, FIGURES, load
from evaluate import RETURNS, SCORES
from forecast import STEP

INK, MUTED, GREY, GRID = "#0b0b0b", "#52514e", "#a8a7a2", "#e6e5e1"
COLOURS = {"arima": "#2a78d6", "ols": "#eb6834", "logit": "#7b5cc4", "gbm": "#3a9b6a", "gbm_p": "#3a9b6a"}
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
                     "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "axes.labelcolor": MUTED,
                     "axes.grid": True, "grid.color": GRID, "axes.axisbelow": True, "legend.frameon": False,
                     "axes.titlelocation": "left", "axes.titlecolor": INK, "axes.titlesize": 12})
SOURCE = "Source: Bitstamp BTC/USD daily candles, public API."


def save(fig, name, title, subtitle, caption):
    fig.text(0.015, 0.985, title, fontsize=15, fontweight="bold", color=INK, va="top")
    fig.text(0.015, 0.925, subtitle, color=MUTED, va="top")
    fig.text(0.015, 0.012, "\n".join(textwrap.wrap(caption, 138)), color=MUTED, fontsize=9, va="bottom")
    fig.savefig(FIGURES / name, dpi=200, facecolor="white")
    plt.close(fig)


def percent(ax):
    fmt = lambda v, _: f"{v * 100:.0f}%".replace("-", "−")
    ax.xaxis.set_major_formatter(fmt)
    ax.yaxis.set_major_formatter(fmt)


def leak_figure(df):
    d = df.loc[leak.WINDOW[0]:leak.WINDOW[1]]
    r = np.log(df.close).diff().loc[d.index].values
    train, test = leak.holdout(len(r))
    fig, axes = plt.subplots(1, 2, figsize=(10, 5.4), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.78, bottom=0.17, wspace=0.08)
    panels = ((axes[0], leak.same_day_returns, "Using that same day's open, high and low"),
              (axes[1], leak.prior_day_returns, "Using only what was known the day before"))
    for ax, inputs, title in panels:
        actual, predicted = leak.fit_score(inputs(df).loc[d.index].values, r, train, test, leak.StandardScaler)
        r2 = 1 - ((actual - predicted) ** 2).sum() / ((actual - actual.mean()) ** 2).sum()
        ax.plot([-0.15, 0.15], [-0.15, 0.15], color=GREY, lw=1)
        ax.scatter(predicted, actual, s=26, color=COLOURS["arima"], alpha=0.7, edgecolor="white", linewidth=0.5)
        ax.set_title(title)
        ax.text(0.04, 0.93, f"R² = {r2:.2f}".replace("-", "−"), transform=ax.transAxes, color=INK, fontsize=12)
        ax.set_xlabel("Predicted daily return")
        percent(ax)
    axes[0].set_ylabel("Actual daily return")
    axes[0].set_xlim(-0.15, 0.15)
    axes[0].set_xticks([-0.1, -0.05, 0, 0.05, 0.1])
    axes[0].set_ylim(-0.17, 0.17)
    save(fig, "leak.png", "The score comes from knowing the day's range",
         f"The same regression on the last {len(test)} of the paper's dates, with price trend removed.",
         "A forecast on the grey line would be exactly right. Both panels train on the earlier 90% of the paper's dates, 21 December 2016 to "
         "21 December 2021, with scaling fit on those days alone. " + SOURCE)


def price_figure(f):
    s = f.iloc[-120:]
    actual, implied, no_change = s.close * np.exp(s.r_next), s.close * np.exp(s.ols), s.close
    dates = s.index + pd.Timedelta(days=1)
    fig, ax = plt.subplots(figsize=(10, 5.4))
    fig.subplots_adjust(left=0.09, right=0.98, top=0.82, bottom=0.17)
    ax.plot(dates, actual, color=INK, lw=2, label="Actual close")
    ax.plot(dates, implied, color=COLOURS["ols"], lw=2, label="Lagged regression, implied price")
    ax.plot(dates, no_change, color=GREY, lw=1.4, ls=(0, (4, 2)), label="No change from the day before")
    ax.yaxis.set_major_formatter(lambda v, _: f"${v:,.0f}")
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.legend(loc="best")
    save(fig, "price.png", "A forecast can track the price and still add nothing",
         "The last 120 days. The regression's implied price follows the no-change line, one day behind the actual close.",
         f"Implied price is the day's close times the exponential of the forecast return. This chart shows the method and says nothing about future prices. {SOURCE}")


def skill_figure(f):
    fig, ax = plt.subplots(figsize=(10, 5.4))
    fig.subplots_adjust(left=0.09, right=0.98, top=0.82, bottom=0.17)
    ax.axhline(0, color=MUTED, lw=1)
    for col, label in RETURNS.items():
        ax.plot(f.index, (f.r_next ** 2 - (f.r_next - f[col]) ** 2).cumsum(), color=COLOURS[col], lw=2, label=label)
    ax.set_ylabel("Running total of squared-error gap")
    ax.legend(loc="best")
    save(fig, "skill.png", "No model pulls ahead of a random walk",
         "Above zero, the model has made smaller squared errors than a no-change forecast so far. Below, larger.",
         f"Next-day log returns, walk-forward, each model refit every {STEP} days on earlier data only. {SOURCE}")


def calibration_figure(f):
    fig, axes = plt.subplots(1, 4, figsize=(10, 4.6), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.74, bottom=0.25, wspace=0.08)
    for ax, (col, label) in zip(axes, SCORES.items()):
        p = f[f"cal_{col}"]
        bins = pd.qcut(p, 8, duplicates="drop")
        table = f.up.groupby(bins, observed=True).agg(["mean", "count"])
        x = p.groupby(bins, observed=True).mean()
        se = np.sqrt(table["mean"] * (1 - table["mean"]) / table["count"])
        ax.plot([0.4, 0.7], [0.4, 0.7], color=GREY, lw=1)
        ax.errorbar(x, table["mean"], yerr=1.96 * se, fmt="o", color=COLOURS[col], ms=6, lw=1.2, mec="white")
        ax.set_title(label, fontsize=11)
        ax.set_xlim(0.4, 0.7)
        ax.set_ylim(0.4, 0.7)
        ax.set_xticks([0.45, 0.55, 0.65])
        percent(ax)
    axes[0].set_ylabel("Share of days that went up")
    fig.supxlabel("Calibrated chance of an up day", y=0.13, color=MUTED, fontsize=11)
    save(fig, "calibration.png", "Forecast chances stay close to the share of up days",
         "Days are split into eight equal groups by forecast. Bars are 95% ranges; on the grey line, forecasts match outcomes.",
         f"Each model's forecasts are turned into probabilities using only its earlier forecasts and outcomes. {SOURCE}")


if __name__ == "__main__":
    FIGURES.mkdir(exist_ok=True)
    f = pd.read_csv(DATA / "scored.csv", parse_dates=["date"], index_col="date")
    leak_figure(load())
    price_figure(f)
    skill_figure(f)
    calibration_figure(f)

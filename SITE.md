# Proposed site text

For whoever adds this to atamalfuzi.github.io. Figures are in `figures/`.

## Portfolio entry

**Checking my Bitcoin forecasts**

Rebuild of an MA paper, 2022 and 2026

My 2022 paper reported a regression score of 0.9994 for Bitcoin's closing price. The rebuild shows that score came from using the same day's high and low, and tests the models again using only earlier information. Across 3,582 days to November 2026, none of them beats a forecast of no change.

## Project page

### Checking my Bitcoin forecasts

In 2022 I wrote an MA paper at SFU comparing ways to forecast Bitcoin's price. It concluded that a linear regression worked best, with a score of 0.9994. I rebuilt it in 2026 to see whether that held up. It didn't, and the reasons are useful for anyone evaluating a forecast. This is a lesson in method. It isn't trading or investment advice.

**Where the score came from.** The regression predicted each day's close from that day's open, high, low and volume. The high and low aren't known until the day is over, so this can't be used as a forecast. Rebuilding the setup from the paper's own data file gives 0.9995, against the 0.9994 reported. On daily returns, the same regression has an R² of 0.79 with same-day inputs and −0.02 with inputs from the day before, using the Bitstamp prices in this chart.

Figure: `figures/leak.png`. Alt text: Two scatter charts of predicted against actual daily returns. With same-day inputs the points follow the diagonal, R² 0.79. With inputs from the day before they form a vertical band around zero, R² −0.02.

**Why fixing that wasn't enough.** Even without the leak, a regression on price levels scores 0.97, and repeating yesterday's price scores the same. Prices are close to their value the day before, so R² on levels says little. The paper also ranked its models by MSE on three different scales: scaled prices, log returns and, for the logit, a log loss. Those numbers can't be compared. On its own scale, the paper's ARIMA was slightly worse than a forecast of no change.

**The rebuild.** Every model now forecasts the next day's log return, using only what was known at the previous close, and is refit on earlier days only. I compared ARIMA, a lagged regression and gradient boosting with a random walk over 3,582 days, from January 2017 to November 2026, using Diebold-Mariano tests.

| Model | Out-of-sample R² vs random walk | Diebold-Mariano p |
|---|---:|---:|
| ARIMA(4,1,1) | −0.12% | 0.73 |
| Lagged regression | −0.58% | 0.13 |
| Gradient boosting | −3.27% | 0.001 |

Figure: `figures/skill.png`. Alt text: Line chart of each model's running squared-error gap against a random walk from 2017 to 2026. All three stay at or below zero. ARIMA and the regression stay close to it, and gradient boosting falls furthest.

ARIMA and the regression can't be told apart from a random walk, and gradient boosting is measurably worse. A no-change forecast still gets an R² of 0.9985 on price levels, close to the paper's number.

Figure: `figures/price.png`. Alt text: Line chart of the last 120 days. The regression's implied price and the no-change forecast overlap, both one day behind the actual close.

**Direction.** The paper's logit had AUCs of 0.54 and 0.51, close to a coin flip, and that was its most reliable result. In the rebuild, after calibrating each model's forecasts on its own past, accuracy is about 51.7% against 51.6% for always saying up. The difference isn't significant.

Figure: `figures/calibration.png`. Alt text: Four small charts of calibrated probability of an up day against the share of days that went up. Forecasts stay between about 50% and 60%, with wide ranges around each point.

**What this does and doesn't show.** The results don't prove that returns can't be forecast: with this many days, a true out-of-sample R² under about 0.7% couldn't be told from zero. They come from one exchange, a daily horizon and a small set of inputs. The repository's tests fail if any input or forecast changes when later prices change.

Code and full results: [github.com/atamalfuzi/btc_forecast_check](https://github.com/atamalfuzi/btc_forecast_check). Daily prices come from Bitstamp's public API and aren't redistributed. Results as of 2 November 2026.

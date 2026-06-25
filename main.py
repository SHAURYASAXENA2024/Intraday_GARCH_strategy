# ============================================================
# PROJECT 3: Intraday Strategy Using GARCH Model
# ============================================================
# Steps:
# 1. Load Simulated Daily and Simulated 5-minute data.
# 2. Fit GARCH model on daily data and predict 1-day ahead volatility.
# 3. Calculate prediction premium and form a daily signal.
# 4. Merge with intraday data and calculate intraday indicators.
# 5. Generate position entry and hold until end of day.
# 6. Calculate final strategy returns.
#
# Required packages:
# pandas, numpy, matplotlib, arch, pandas_ta, os
# ============================================================

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from arch import arch_model
import pandas_ta
import pandas as pd
import numpy as np
import os


# ── 1. Load Simulated Daily and 5-minute data ───────────────────────────────

data_folder = "C:/Users/user/Desktop/Python Scripts"  # <-- Change to your path

daily_df = pd.read_csv(os.path.join(data_folder, "simulated_daily_data.csv"))

daily_df = daily_df.drop("Unnamed: 7", axis=1)

daily_df["Date"] = pd.to_datetime(daily_df["Date"])

daily_df = daily_df.set_index("Date")


intraday_5min_df = pd.read_csv(os.path.join(data_folder, "simulated_5min_data.csv"))

intraday_5min_df = intraday_5min_df.drop("Unnamed: 6", axis=1)

intraday_5min_df["datetime"] = pd.to_datetime(intraday_5min_df["datetime"])

intraday_5min_df = intraday_5min_df.set_index("datetime")

intraday_5min_df["date"] = pd.to_datetime(intraday_5min_df.index.date)

print(intraday_5min_df)


# ── 2. Fit GARCH model and predict 1-day ahead volatility ──────────────────

daily_df["log_ret"] = np.log(daily_df["Adj Close"]).diff()

daily_df["variance"] = daily_df["log_ret"].rolling(180).var()

daily_df = daily_df["2020":]


def predict_volatility(x):
    best_model = arch_model(y=x, p=1, q=3).fit(update_freq=5, disp="off")

    variance_forecast = best_model.forecast(horizon=1).variance.iloc[-1, 0]

    print(x.index[-1])

    return variance_forecast


daily_df["predictions"] = (
    daily_df["log_ret"].rolling(180).apply(lambda x: predict_volatility(x))
)

daily_df = daily_df.dropna()

print(daily_df)


# ── 3. Calculate prediction premium and form daily signal ──────────────────

daily_df["prediction_premium"] = (
    daily_df["predictions"] - daily_df["variance"]
) / daily_df["variance"]

daily_df["premium_std"] = daily_df["prediction_premium"].rolling(180).std()

daily_df["signal_daily"] = daily_df.apply(
    lambda x: (
        1
        if (x["prediction_premium"] > x["premium_std"])
        else (-1 if (x["prediction_premium"] < x["premium_std"] * -1) else np.nan)
    ),
    axis=1,
)

# Shift signal by 1 day to avoid lookahead bias
daily_df["signal_daily"] = daily_df["signal_daily"].shift()

print(daily_df)

plt.style.use("ggplot")
daily_df["signal_daily"].plot(kind="hist")
plt.show()


# ── 4. Merge with intraday data and calculate intraday indicators ───────────

final_df = (
    intraday_5min_df.reset_index()
    .merge(daily_df[["signal_daily"]].reset_index(), left_on="date", right_on="Date")
    .drop(["date", "Date"], axis=1)
    .set_index("datetime")
)

final_df["rsi"] = pandas_ta.rsi(close=final_df["close"], length=20)

final_df["lband"] = pandas_ta.bbands(close=final_df["close"], length=20).iloc[:, 0]

final_df["uband"] = pandas_ta.bbands(close=final_df["close"], length=20).iloc[:, 2]

final_df["signal_intraday"] = final_df.apply(
    lambda x: (
        1
        if (x["rsi"] > 70) & (x["close"] > x["uband"])
        else (-1 if (x["rsi"] < 30) & (x["close"] < x["lband"]) else np.nan)
    ),
    axis=1,
)

final_df["return"] = np.log(final_df["close"]).diff()

print(final_df)


# ── 5. Generate position entry and hold until end of day ───────────────────

# Signal logic:
# If daily signal = 1 (high volatility expected) AND intraday overbought → SHORT
# If daily signal = -1 (low volatility expected) AND intraday oversold  → LONG

final_df["return_sign"] = final_df.apply(
    lambda x: (
        -1
        if (x["signal_daily"] == 1) & (x["signal_intraday"] == 1)
        else (1 if (x["signal_daily"] == -1) & (x["signal_intraday"] == -1) else np.nan)
    ),
    axis=1,
)

# Hold position until end of day using forward fill within each day
final_df["return_sign"] = final_df.groupby(pd.Grouper(freq="D"))[
    "return_sign"
].transform(lambda x: x.ffill())

final_df["forward_return"] = final_df["return"].shift(-1)

final_df["strategy_return"] = final_df["forward_return"] * final_df["return_sign"]

daily_return_df = final_df.groupby(pd.Grouper(freq="D"))["strategy_return"].sum()


# ── 6. Calculate and visualize final strategy returns ──────────────────────

strategy_cumulative_return = np.exp(np.log1p(daily_return_df).cumsum()).sub(1)

strategy_cumulative_return.plot(figsize=(16, 6))

plt.title("Intraday Strategy Returns")

plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1))

plt.ylabel("Return")

plt.show()

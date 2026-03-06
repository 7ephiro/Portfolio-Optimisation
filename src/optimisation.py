# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: .venv (3.12.6)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Grab Stock Data

# %% [markdown]
# ## 1. Stock Data

# %%
#Grab Data
import yfinance as yf

#Usual Suspects
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns

# Portfolio 
symbols = [
    # Banks
    'HSBA.L',
    'BARC.L',
    'LLOY.L',

    # Insurance
    'AV.L',
    'LGEN.L',
    'PRU.L',

    # Mining
    'RIO.L',
    'AAL.L',
    'ANTO.L',

    # Healthcare
    'AZN.L',
    'GSK.L',
    'SN.L',

    # Consumer Staples
    'ULVR.L',
    'DGE.L',
    'IMB.L'
]

#Get the stock data
data = yf.download(symbols, start="2019-01-01", end="2025-12-31", auto_adjust=False)
if data is None:
    raise RuntimeError("yfinance returned no data")

data.head()

# %% [markdown]
# ## 2. Indice Data

# %%
#Import FTSE 100 data and plot cum_returns for performance comparison
ftse_index = yf.download('^FTSE', start='2019-01-01', end="2025-12-31", auto_adjust=False)
if ftse_index is None:
    raise RuntimeError("yfinance returned no FTSE 100 data")

ftse_index.head()

# %% [markdown]
# # Visualise Return

# %%
#Visualize cumulative returns of each stock in the portfolio
if data is None:
    raise RuntimeError("Stock data is missing")

close_prices = data["Adj Close"]
portfolio_returns = close_prices.pct_change().dropna()

port_comps_rets_cumprod = ((portfolio_returns + 1).cumprod() - 1) * 100

#Plot
fig = px.line(port_comps_rets_cumprod, x=port_comps_rets_cumprod.index, y=port_comps_rets_cumprod.columns, title='Cumulative Returns of Portfolio Stocks (2019-2025)')

fig.update_xaxes(title_text='Date')
fig.update_yaxes(title_text='Cumulative Return in %')

fig.show()


# %% [markdown]
# # Optimization

# %% [markdown]
# ## 1. Train Test Split Data

# %%
from pypfopt import risk_models, expected_returns
from pypfopt.expected_returns import ema_historical_return
from pypfopt.risk_models import exp_cov
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.plotting import plot_efficient_frontier
from pypfopt.plotting import plot_weights
from pypfopt.cla import CLA

#Train Test Split the Data
train = portfolio_returns["2019-01-01":"2023-12-31"]
test = portfolio_returns["2024-01-01":"2025-12-31"]

#Get mu and Sigma from expected_returns and risk
mu = expected_returns.ema_historical_return(train, returns_data = True, span = 500)
Sigma = risk_models.exp_cov(train, returns_data = True, span = 180)


# %% [markdown]
# ## 2. Efficient Frontier

# %%
#For visual purposes and calculate efficient frontier
from numbers import Real


def as_real_float(value: object, label: str) -> float:
    if value is None or not isinstance(value, Real):
        raise RuntimeError(f"{label} is not a valid numeric value")
    return float(value)


rf = 0.0375
solver = "SCS"
solver_options = {"max_iters": 300000, "eps": 1e-6}

# Build frontier with CLA
cla = CLA(mu, Sigma)
ret_ef_raw, vol_ef_raw, _ = cla.efficient_frontier(points=500)
ret_ef = [as_real_float(v, "Frontier return") for v in ret_ef_raw]
vol_ef = [as_real_float(v, "Frontier volatility") for v in vol_ef_raw]

#Calculate the minimum volatility
ef = EfficientFrontier(mu, Sigma, solver=solver, solver_options=solver_options)
ef.min_volatility()
min_perf = ef.portfolio_performance(risk_free_rate=rf)
min_vol_ret = as_real_float(min_perf[0], "Minimum-variance return")
min_vol_vol = as_real_float(min_perf[1], "Minimum-variance volatility")

#Calculate the maximum sharpe ratio
ef = EfficientFrontier(mu, Sigma, solver=solver, solver_options=solver_options)
ef.max_sharpe(risk_free_rate=rf)
max_perf = ef.portfolio_performance(risk_free_rate=rf)
max_sharpe_ret = as_real_float(max_perf[0], "Max-Sharpe return")
max_sharpe_vol = as_real_float(max_perf[1], "Max-Sharpe volatility")

#Calculate equal weight portfolio
n_assets = len(mu)
ew_weights = np.repeat(1 / n_assets, n_assets)
ew_ret = float(np.dot(ew_weights, np.asarray(mu)))
ew_var = float(np.dot(ew_weights, np.asarray(Sigma) @ ew_weights))
ew_vol = float(np.sqrt(max(ew_var, 0.0)))

#Plot
sns.set_theme()

fig, ax = plt.subplots(figsize = [15,10])

sns.lineplot(x=vol_ef, y=ret_ef, label="Efficient Frontier", ax=ax)
sns.scatterplot(x=[min_vol_vol], y=[min_vol_ret], ax=ax, label="Minimum Variance Portfolio", color="purple", s=100)
sns.scatterplot(x=[max_sharpe_vol], y=[max_sharpe_ret], ax=ax, label="Maximum Sharpe Portfolio", color="green", s=100)
sns.scatterplot(x=[ew_vol], y=[ew_ret], ax=ax, label="Equal Weight Portfolio", color="orange", s=100)

x_cml_max = max(vol_ef) * 1.05
y_cml_max = rf + ((max_sharpe_ret - rf) / max_sharpe_vol) * x_cml_max
sns.lineplot(x=[0.0, x_cml_max], y=[rf, y_cml_max], label="Capital Market Line", ax=ax, color="r")

ax.set_xlabel("Volatility")
ax.set_ylabel("Mean Return")
plt.legend(fontsize='large')
plt.title("Efficient Frontier", fontsize='20')


# %% [markdown]
# # Weights

# %% [markdown]
# ## 1. Minimum Variance

# %%
# Minimum Variance
ef = EfficientFrontier(mu, Sigma, solver="SCS", solver_options={"max_iters": 300000, "eps": 1e-6})
raw_weights_minvar_exp = ef.min_volatility()

plot_weights(raw_weights_minvar_exp)
ef.portfolio_performance(verbose = True, risk_free_rate = 0.0375)


# %% [markdown]
# ## 2. Maximum Sharpe

# %%
# Maximum Sharpe
ef = EfficientFrontier(mu, Sigma, solver="SCS", solver_options={"max_iters": 300000, "eps": 1e-6})
raw_weights_maxsharpe_exp = ef.max_sharpe(risk_free_rate=0.0375)

plot_weights(raw_weights_maxsharpe_exp)
ef.portfolio_performance(verbose = True, risk_free_rate = 0.0375)


# %% [markdown]
# ## 3. Equal Weight

# %%
# Equal Weight
n_assets = len(raw_weights_minvar_exp)
equal_weight = 1 / n_assets
raw_weights_equal_exp = {ticker: equal_weight for ticker in raw_weights_minvar_exp}

plot_weights(raw_weights_equal_exp)
print(f"Equal weight per asset: {equal_weight:.4%}")

ew_weights = np.asarray(list(raw_weights_equal_exp.values()), dtype=float)
ew_ret = float(np.dot(ew_weights, np.asarray(mu)))
ew_var = float(np.dot(ew_weights, np.asarray(Sigma) @ ew_weights))
ew_vol = float(np.sqrt(max(ew_var, 0.0)))
rf_local = 0.0375
ew_sharpe = (ew_ret - rf_local) / ew_vol if ew_vol > 0 else float('nan')

print(f"Expected annual return: {ew_ret:.2%}")
print(f"Annual volatility: {ew_vol:.2%}")
print(f"Sharpe Ratio: {ew_sharpe:.2f}")


# %% [markdown]
# # Backtesting

# %%
#List the weights of each portfolio mix & set them into variables
weights_minvar_exp = list(raw_weights_minvar_exp.values())
weights_maxsharpe_exp = list(raw_weights_maxsharpe_exp.values())
weights_equal_exp = list(raw_weights_equal_exp.values())

ret_1 = ((test.dot(weights_minvar_exp) + 1).cumprod() - 1) * 100
ret_2 = ((test.dot(weights_maxsharpe_exp) + 1).cumprod() - 1) * 100
ret_3 = ((test.dot(weights_equal_exp) + 1).cumprod() - 1) * 100

#Include the FTSE 100 for returns comparison
if ftse_index is None:
    raise RuntimeError("FTSE 100 data is missing")
ftse_close = ftse_index.loc["2024-01-01":"2025-12-31", "Adj Close"]
ind_ret = ((ftse_close.pct_change() + 1).cumprod() - 1) * 100

#Set everything into a dataframe
back = pd.DataFrame({"MinVar":ret_1, "MaxSharpe":ret_2, "EqualWeight":ret_3})
back = pd.concat([back, ind_ret],  join = "outer", axis = 1, sort = True)
back.drop(back.tail(1).index,inplace=True)

back.interpolate(method = "linear", inplace = True)

#Plot
fig = px.line(back, x = back.index, y = back.columns, title = "Portfolio Performance")
fig.update_xaxes(title_text='Date')
fig.update_yaxes(title_text='Cumulative Return in %')

fig.show()

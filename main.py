import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from IPython.display import display
from yahoo_fin import options
from yahoo_fin import stock_info as si

import BlackScholes as bs
import Merton as m
import Kou as k
import VarianceGamma as vg

# import scipy.stats as stats
# import math

np.random.seed(27)

# %%%%%%%%%%%%%%%%%%%%%%%%       Get Option Data       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
symbol = 'META'  # AAPL, TSLA, MSFT, META, CAT, STM, GOOG, NFLX, AMZN
dates = options.get_expiration_dates(symbol)
print(dates)

T_str = 'June 21, 2024'  # This is the first "greater than a year" expiry date
calls = options.get_calls(symbol, T_str)
puts = options.get_puts(symbol, T_str)

T_datetime = datetime.strptime(T_str, '%B %d, %Y')
ttm = (T_datetime - datetime.now()).days / 365.0  # to use the time to maturity in float mode

calls['Time-to-maturity'] = ttm
puts['Time-to-maturity'] = ttm
display(calls.columns)

print(calls.head())

# historic volatility
stock_data = si.get_data(symbol, start_date='31/05/2021', end_date='31/05/2023')
print(stock_data.head())
stock_data['Returns'] = stock_data['close'] / stock_data['close'].shift()
stock_data['Log Returns'] = np.log(stock_data['Returns'])
print(stock_data.head())
volatility = stock_data['Log Returns'].std() * np.sqrt(252)
print(f'Historical volatility: {round(volatility, 3)}')


# %%%%%%%%%%%%%%%%%%%%%%%%       Data preparation      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# print(calls.dtypes)
# print(puts.dtypes)
# convert the 'Implied Volatility' column to a numeric type
calls['Implied Volatility'] = pd.to_numeric(calls['Implied Volatility'].str.strip('%')) / 100
puts['Implied Volatility'] = pd.to_numeric(puts['Implied Volatility'].str.strip('%')) / 100
# choose one option as starting point
option = calls.iloc[np.random.randint(len(calls))]
print(option)

# FIXED PARAMETERS (used for all models)
S0 = si.get_live_price(symbol)  # get live price of stock
print(f'{symbol} Current stock price: {round(S0, 2)}')
T = ttm  # Expiry Date in years
days = 252
paths = 1000
K = option[2]  # Strike price
sigma = volatility
r = 0.05  # risk.free interest rate
size = (days, paths)

# %%%%%%%%%%%%%%%%%%%%%%%%        Simulate paths       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

SBlackScholes = bs.BlackScholesPath(T, days, paths, sigma, r, S0)

lamda = 0.5
jump_mean = 0
jump_std = 0.2
SMerton = m.MertonPath(T, days, paths, sigma, r, lamda, jump_mean, jump_std, S0)

eta_1 = 10  # upward jump magnitude
eta_2 = 2  # downward jump
p = 0.4  # q = 0.6
SKou = k.KouPath(T, days, paths, sigma, r, lamda, p, eta_1, eta_2, S0)

theta = 0.15
nu = 0.7
SVarGamma = vg.VarianceGammaPath1(T, days, paths, 0.15, 0.1, nu, theta, S0)
SVarGamma2 = vg.VarianceGammaPath2(T, days, paths, 0.15, 0.1, nu, theta, S0)
# %%%%%%%%%%%%%%%%%%%%%%%%       Visualize paths       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

##################       BLACK AND SCHOLES MODEL       ##################
bs.plotBSPath(SBlackScholes, symbol)  # Plot all paths

subset = 5  # Plot a subset of 5
SBS_subset = np.empty((subset, days))
random_indices = np.random.choice(paths, size=subset)
for path in range(subset):
    SBS_subset[path] = SBlackScholes[:, random_indices[path]]
bs.plotBSPath(SBS_subset.T, symbol)

bs.plotBSPath(SBlackScholes[:, np.random.choice(paths)], symbol)  # Plot only one


##################       MERTON JUMP DIFFUSION       ##################
m.plotMertonPath(SMerton, symbol)  # Plot all paths

subset = 5  # Plot a subset of 5
SM_subset = np.empty((subset, days))
random_indices = np.random.choice(paths, size=subset)
for path in range(subset):
    SM_subset[path] = SMerton[:, random_indices[path]]
m.plotMertonPath(SM_subset.T, symbol)

m.plotMertonPath(SMerton[:, np.random.choice(paths)], symbol)  # Plot only one


##################       KOU JUMP DIFFUSION       ##################
# Plot all paths
k.plotKouPath(SKou, symbol)

# Plot a subset of 5
subset = 5
SK_subset = np.empty((subset, days))
random_indices = np.random.choice(paths, size=subset)
for path in range(subset):
    SK_subset[path] = SKou[:, random_indices[path]]
k.plotKouPath(SK_subset.T, symbol)

# Plot only one
k.plotKouPath(SKou[:, np.random.choice(paths)], symbol)


##################       VARIANCE GAMMA PROCESS       ##################
method = ['Time changed BM', 'Difference of Gammas']

# Plot all paths
vg.plotVGPath(SVarGamma, symbol, method[0])
vg.plotVGPath(SVarGamma2, symbol, method[1])

# Plot a subset of 5
subset = 5  # Plot a subset of 5
SVG_subset = np.empty((subset, days))
random_indices = np.random.choice(paths, size=subset)
for path in range(subset):
    SVG_subset[path] = SVarGamma[:, random_indices[path]]
vg.plotVGPath(SVG_subset.T, symbol, method[0])

subset = 5
SVG_subset = np.empty((subset, days))
random_indices = np.random.choice(paths, size=subset)
for path in range(subset):
    SVG_subset[path] = SVarGamma2[:, random_indices[path]]
vg.plotVGPath(SVG_subset.T, symbol, method[1])

# Plot only one
vg.plotVGPath(SVarGamma[:, np.random.choice(paths)], symbol, method[0])
vg.plotVGPath(SVarGamma[:, np.random.choice(paths)], symbol, method[1])


# %%%%%%%%%%%%%%%%%%%%%%%%    Visualize distributions  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Average prices
bs.plotBSDist(SBlackScholes, symbol)
m.plotMertonDist(SMerton, symbol)
k.plotKouDist(SKou, symbol)
vg.plotVGDist(SVG_subset, symbol)

# Log returns ALL PATHS
bs.plotBSLogReturns(SBlackScholes, symbol)
m.plotMertonLogReturns(SMerton, symbol)
k.plotKouLogReturns(SKou, symbol)
vg.plotVGLogReturns(SVarGamma2, symbol)

# Log returns ONE RANDOM
bs.plotBSLogReturns(SBlackScholes[:, np.random.choice(paths)], symbol)
m.plotMertonLogReturns(SMerton[:, np.random.choice(paths)], symbol)
k.plotKouLogReturns(SKou[:, np.random.choice(paths)], symbol)
vg.plotVGLogReturns(SVarGamma2[:, np.random.choice(paths)], symbol)

# %%%%%%%%%%%%%%%%%%%%%%%%    Parameters estimation   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Estimate params of VG (ref. Seneta 2004)
sigma_est = np.sqrt(vg.find_moment(2, theta, nu, sigma))
theta_est = sigma*(vg.find_moment(3,theta, nu, sigma))/(3*nu)
nu_est = (vg.find_moment(4, theta, nu, sigma)/3) - 1

print(f'Variance Gamma Estimated Params\n\t\tSTART\tEXT')
print(f'Sigma:  {round(sigma,2)} -> {round(sigma_est,4)}')
print(f'Theta: {theta} -> {round(theta_est,4)}')
print(f'Nu:     {nu} -> {round(nu_est,4)}')

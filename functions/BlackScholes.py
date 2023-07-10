# %%%%%%%%%%%%%%%%%%%%%%%       Black and Scholes Model       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""S_t = S_0 exp((mu-0.5*sigma^2)*t + sigma*sqr(t)Z*(t)

     S_t: the spot price of the underlying asset at time t
     r: the risk-free interest rate
     sigma: the implied volatility of returns of the underlying asset
     K: the strike price of the option
     C0: the price of a European call option at time t = 0
     t: current time in years
     T: expiry date
     N: number of simulated paths
"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as ss


def BlackScholesPath(T, days, N, sigma, r, S0):
    S = np.zeros((days, N))
    S[0] = S0
    dt = T / days
    for t in range(1, days):
        Z = np.random.normal(size=N)
        S[t] = S[t - 1] * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z)
    return S


# plot the price paths
def plotBSPath(Spaths, symbol, ax=None):
    if ax is None:
        ax = plt.gca()
    plt.figure(figsize=(10, 6))
    ax.plot(Spaths)
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Price')
    ax.set_title(f'B&S Model Price Paths for {symbol}')
    return


# plot the distribution of prices average
def plotBSDist(Spaths, symbol, ax=None):
    if ax is None:
        ax = plt.gca()
    avg_path = np.mean(Spaths, axis=0)
    ax.hist(avg_path, bins='auto')
    ax.set_xlabel('price')
    ax.set_ylabel('frequency')
    ax.set_title(f'B&S model: Lognormal distribution of {symbol} prices')
    return


def plotBSLogReturns(Spaths, symbol, ax=None):
    if ax is None:
        ax = plt.gca()
    log_returns = np.log(Spaths[1:] / Spaths[:-1])
    log_returns = log_returns.flatten()
    sns.kdeplot(log_returns, label='Log Returns', ax=ax)
    ax.set_xlabel('Log Return')
    ax.set_ylabel('Density')
    ax.set_title(f'Black and Scholes')
    ax.legend()
    return


def plotBSAtFixedTime(SBlackScholes, time, symbol, ax):
    if ax is None:
        ax = plt.gca()
    fixed_values = SBlackScholes[time, :]

    # Plotting the histogram
    hist, bins, _ = ax.hist(fixed_values, bins=30, density=True, alpha=0.7, label='Histogram')

    # Plotting the KDE approx
    sns.kdeplot(fixed_values, color='r', label='Approximation', ax=ax)

    ax.set_xlabel(f'{symbol} price after T = {time+1} days')
    ax.set_ylabel('Probability Density')
    ax.set_title(f'Black-Scholes Price at T ={time + 1}')
    ax.grid(True)
    ax.legend()
    return

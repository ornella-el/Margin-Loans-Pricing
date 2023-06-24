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


def BlackScholesPath(T, days, N, sigma, r, S0):
    S = np.zeros((days, N))
    S[0] = S0
    dt = T / days
    for t in range(1, days):
        Z = np.random.normal(size=N)
        S[t] = S[t - 1] * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z)
    return S


# plot the price paths
def plotBSPath(Spaths, symbol):
    plt.figure(figsize=(10, 6))
    plt.plot(Spaths)
    plt.xlabel('Time (days)')
    plt.ylabel('Price')
    plt.title(f'B&S Model Price Paths for {symbol}')
    plt.show()
    return


# plot the distribution of prices average
def plotBSDist(Spaths, symbol):
    avg_path = np.mean(Spaths, axis=0)
    plt.hist(avg_path, bins='auto')
    plt.xlabel('price')
    plt.ylabel('frequency')
    plt.title(f'B&S model: Lognormal distribution of {symbol} prices')
    plt.show()
    return


def plotBSLogReturns(Spaths, symbol):
    log_returns = np.log(Spaths[1:] / Spaths[:-1])
    log_returns = log_returns.flatten()
    sns.kdeplot(log_returns, label='Log Returns')
    plt.xlabel('Log Return')
    plt.ylabel('Density')
    plt.title(f'Distribution of BS Log Returns for {symbol}')
    plt.legend()
    plt.show()
    return

